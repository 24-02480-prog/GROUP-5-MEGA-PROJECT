from __future__ import annotations

import pandas as pd

from src.evaluate import evaluate_scenarios, rate_error_row
from src.rate_utils import compare_to_reference
from src.scenarios import apply_scenario
from src.schema import validate_feature_table
from src.synthetic_data import generate_synthetic_record
from src.t01_loader import load_records
from src.t02_ecg import estimate_ecg_hr
from src.t03_ppg import estimate_ppg_pr
from src.t04_resp import estimate_resp_rate
from src.t05_quality import assess_signal_quality
from src.t06_fusion import fuse_windows


def _rates_and_quality(variant):
    rates = [
        estimate_ecg_hr(variant),
        estimate_ppg_pr(variant),
        estimate_resp_rate(variant),
    ]
    quality = {
        name: assess_signal_quality(variant, name)
        for name in ("II", "PLETH", "RESP")
    }
    return rates, quality


def test_clean_fusion_is_normal_and_spo2_has_reference_lineage(waveform):
    variant = apply_scenario(waveform, "A")
    rates, quality = _rates_and_quality(variant)
    timeline, feature_table = fuse_windows(
        variant, rates, quality, scenario="A"
    )
    validate_feature_table(feature_table)
    assert timeline["demo_flag"].eq("NORMAL-DEMO").all()
    assert timeline["data_source"].eq("SYNTHETIC").all()
    assert timeline["SpO2_ref_lineage"].eq(
        "provided_reference:Numerics.SpO2"
    ).all()


def test_fusion_gates_non_ok_windows_and_threshold_override(waveform):
    variant = apply_scenario(waveform, "C")
    rates, quality = _rates_and_quality(variant)
    timeline, feature_table = fuse_windows(
        variant, rates, quality, scenario="C"
    )
    assert timeline["demo_flag"].eq("NO-DATA").all()
    assert timeline[["HR", "PR", "RR"]].isna().all().all()
    validate_feature_table(feature_table)

    clean = apply_scenario(waveform, "A")
    clean_rates, clean_quality = _rates_and_quality(clean)
    caution, _ = fuse_windows(
        clean,
        clean_rates,
        clean_quality,
        thresholds={"hr_max_bpm": 70.0},
        demo_case="TEST-THRESHOLD-OVERRIDE",
        simulated_flag=True,
    )
    assert caution["demo_flag"].eq("CAUTION-DEMO").all()


def test_rate_error_summary_includes_before_and_after_metrics(waveform):
    variant = apply_scenario(waveform, "A")
    quality = assess_signal_quality(variant, "II")
    estimate = estimate_ecg_hr(variant)
    row = rate_error_row(
        waveform, variant, "HR", "HR", "II", estimate, quality
    )
    assert row["data_source"] == "SYNTHETIC"
    assert row["windows_evaluated_before"] == 16
    assert row["windows_evaluated_after"] == 16
    assert row["MAE_before"] < 0.1


def test_evaluation_exports_error_table_and_feature_figures(waveform, tmp_path):
    errors = evaluate_scenarios([waveform], tmp_path)
    assert len(errors) == 9
    assert errors["data_source"].eq("SYNTHETIC").all()
    assert (tmp_path / "error_table.csv").is_file()
    for filename in (
        "rate_error_summary.png",
        "bland_altman_HR.png",
        "bland_altman_PR.png",
        "bland_altman_RR.png",
        "error_vs_quality_label.png",
        "coverage_vs_mae.png",
        "ecg_signal.png",
        "ppg_signal.png",
        "respiration_signal.png",
    ):
        assert (tmp_path / filename).is_file()
    features = pd.read_csv(tmp_path / "final_feature_table.csv")
    validate_feature_table(features)


def test_real_source_keeps_provenance_in_feature_record_id(tmp_path, monkeypatch):
    from src import t01_loader

    for record_number in (1, 2):
        source = generate_synthetic_record(record_number)
        source.signals.to_csv(
            tmp_path / f"bidmc_{record_number:02d}_Signals.csv", index=False
        )
        source.numerics.to_csv(
            tmp_path / f"bidmc_{record_number:02d}_Numerics.csv", index=False
        )
    monkeypatch.setattr(t01_loader, "DATA_DIR", tmp_path)
    waveforms = load_records("real")
    waveform = waveforms[0]
    rates, quality = _rates_and_quality(apply_scenario(waveform, "A"))
    timeline, feature_table = fuse_windows(
        waveform, rates, quality, scenario="A"
    )
    comparison = compare_to_reference(waveforms, rates)
    assert feature_table["record_id"].eq("real-01").all()
    assert timeline["data_source"].eq("REAL").all()
    assert comparison["data_source"].eq("REAL").all()
