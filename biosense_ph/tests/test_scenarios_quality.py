from __future__ import annotations

import numpy as np

from src.scenarios import apply_quality_example, apply_scenario
from src.t05_quality import assess_quality, assess_signal_quality, window_quality


def test_scenarios_are_deterministic_and_do_not_mutate_source(waveform):
    original = waveform.signals["II"].copy()
    first = apply_scenario(waveform, "B")
    second = apply_scenario(waveform, "B")
    np.testing.assert_array_equal(first.signals["II"], second.signals["II"])
    np.testing.assert_array_equal(waveform.signals["II"], original)
    assert first.simulated_flag is True
    assert apply_scenario(waveform, "A").simulated_flag is False


def test_scenario_c_injects_flatline_and_missing_regions(waveform):
    variant = apply_scenario(waveform, "C")
    values = variant.signals["II"]
    assert np.isnan(values).mean() > 0.2
    assert np.unique(values[int(5 * waveform.fs) : int(15 * waveform.fs)]).size == 1


def test_independent_noise_and_spike_examples_are_simulated(waveform):
    noise = apply_quality_example(waveform, "noise-only")
    spikes = apply_quality_example(waveform, "spikes-only")
    assert noise.scenario == spikes.scenario == "B"
    assert noise.simulated_flag and spikes.simulated_flag
    assert not np.array_equal(noise.signals["II"], spikes.signals["II"])


def test_quality_rules_assign_low_to_flatline_and_missing_to_nan():
    flat = window_quality(np.zeros(30 * 125), 125.0, "II")
    missing_values = np.zeros(30 * 125)
    missing_values[: int(8 * 125)] = np.nan
    missing = window_quality(missing_values, 125.0, "II")
    assert flat["quality"] == "LOW"
    assert missing["quality"] == "MISSING"


def test_quality_table_has_schema_and_clean_synthetic_quality(waveform):
    result = assess_quality(waveform)
    assert len(result) == 16
    assert result["feature"].eq("quality").all()
    assert result["quality"].eq("OK").all()


def test_per_signal_scenario_quality_flags_corruption(waveform):
    variant = apply_scenario(waveform, "B")
    result = assess_signal_quality(variant, "PLETH")
    assert result["quality"].eq("LOW").all()
    assert result["data_source"].eq("SYNTHETIC").all()
