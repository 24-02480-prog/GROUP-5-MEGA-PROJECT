from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.config import get_config
from src.rate_utils import estimate_rate_diagnostics
from src.scenarios import apply_scenario, severity_scenarios
from src.schema import FEATURE_COLUMNS, validate_feature_table
from src.t02_ecg import estimate_ecg_hr
from src.t03_ppg import estimate_ppg_pr
from src.t04_resp import estimate_resp_rate
from src.t05_quality import assess_signal_quality
from src.t06_fusion import fuse_windows
from src.t09_ml_rate import evaluate_rate_correction


def test_severity_scenarios_have_configured_levels_and_are_repeatable(waveform):
    scenarios = severity_scenarios()
    expected = {
        "noise_snr_20db",
        "noise_snr_10db",
        "noise_snr_5db",
        "noise_snr_0db",
        "spike_burst_5",
        "spike_burst_20",
        "spike_burst_50",
        "dropout_10pct",
        "dropout_30pct",
        "dropout_60pct",
        "baseline_drift_low",
        "baseline_drift_high",
        "attenuation_50pct",
        "attenuation_20pct",
    }
    assert set(scenarios) == expected
    for scenario in expected:
        first = apply_scenario(waveform, scenario)
        second = apply_scenario(waveform, scenario)
        np.testing.assert_allclose(
            first.signals["II"], second.signals["II"], equal_nan=True
        )
        assert first.simulated_flag is True


def test_noise_snr_and_dropout_levels_change_signal_by_configured_severity(waveform):
    clean = waveform.signals["II"]
    standard_deviations = []
    for snr in (20, 10, 5, 0):
        noisy = apply_scenario(waveform, f"noise_snr_{snr}db").signals["II"]
        standard_deviations.append(float(np.std(noisy - clean)))
    assert standard_deviations == sorted(standard_deviations)

    for percentage in (10, 30, 60):
        dropped = apply_scenario(
            waveform, f"dropout_{percentage}pct"
        ).signals["II"][: int(30 * waveform.fs)]
        assert np.isclose(np.isnan(dropped).mean(), percentage / 100, atol=0.002)


def test_spike_counts_and_amplitude_attenuation_levels(waveform):
    clean = waveform.signals["II"][: int(30 * waveform.fs)]
    for count in (5, 20, 50):
        spiked = apply_scenario(
            waveform, f"spike_burst_{count}"
        ).signals["II"][: int(30 * waveform.fs)]
        assert np.count_nonzero(spiked != clean) == count
    for percentage, factor in ((50, 0.5), (20, 0.2)):
        attenuated = apply_scenario(
            waveform, f"attenuation_{percentage}pct"
        ).signals["II"][: int(30 * waveform.fs)]
        ratio = np.std(attenuated) / np.std(clean)
        assert np.isclose(ratio, factor, atol=0.01)


def test_rate_diagnostics_include_ibi_and_welch_crosscheck(waveform, monkeypatch):
    import src.rate_utils as rate_utils

    config = deepcopy(get_config())
    config["rates"]["ecg"]["spectral_disagreement_bpm"] = 0.1
    monkeypatch.setattr(rate_utils, "get_config", lambda: config)
    diagnostics = estimate_rate_diagnostics(waveform, "II")
    assert all(np.isfinite(row["ibi_std"]) for row in diagnostics)
    assert all(np.isfinite(row["spectral_rate"]) for row in diagnostics)
    assert all(row["quality"] == "LOW" for row in diagnostics)
    assert estimate_ecg_hr(waveform)["quality"].eq("LOW").all()
    assert estimate_ppg_pr(waveform)["quality"].isin({"OK", "LOW"}).all()
    assert estimate_resp_rate(waveform)["quality"].isin({"OK", "LOW"}).all()


def test_hr_pr_crosscheck_adds_difference_feature_and_low_quality(waveform):
    signals = dict(waveform.signals)
    signals["PLETH"] = np.sin(2 * np.pi * 2.0 * waveform.time_s)
    mismatched = replace(waveform, signals=signals)
    rates = [
        estimate_ecg_hr(mismatched),
        estimate_ppg_pr(mismatched),
        estimate_resp_rate(mismatched),
    ]
    signal_quality = {
        name: assess_signal_quality(mismatched, name)
        for name in ("II", "PLETH", "RESP")
    }
    timeline, feature_table = fuse_windows(
        mismatched, rates, signal_quality, scenario="A"
    )
    difference = feature_table[feature_table["feature"] == "HR_PR_diff"]
    validate_feature_table(feature_table)
    assert feature_table.columns.tolist() == FEATURE_COLUMNS
    assert (difference["value"] > get_config()["cross_check"]["hr_pr_max_difference_bpm"]).all()
    assert difference["quality"].eq("LOW").all()
    assert timeline["HR_PR_diff_quality"].eq("LOW").all()
    assert timeline["overall_quality"].eq("LOW").all()


def test_ml_rate_correction_is_record_held_out_and_schema_valid(two_waveforms, tmp_path):
    metrics, predictions, corrected, peak_based = evaluate_rate_correction(
        two_waveforms, tmp_path
    )
    assert len(metrics) == 6
    assert (
        metrics["training_record"].to_numpy()
        != metrics["test_record"].to_numpy()
    ).all()
    assert set(metrics["corrected_method"]) == {"ML-corrected (Ridge)"}
    assert set(metrics["peak_based_method"]) == {
        "peak-based (median-filtered IBI)"
    }
    assert predictions["training_record"].ne(predictions["record_id"]).all()
    validate_feature_table(corrected)
    validate_feature_table(peak_based)
    assert corrected.columns.tolist() == FEATURE_COLUMNS
    assert (tmp_path / "ml_rate_correction_metrics.csv").is_file()
    assert (tmp_path / "ml_corrected_feature_table.csv").is_file()
