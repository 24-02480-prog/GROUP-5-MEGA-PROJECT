from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.t01_loader import Waveform
from src.t02_ecg import estimate_ecg_hr
from src.t03_ppg import estimate_ppg_pr
from src.t04_resp import estimate_resp_rate


def test_ecg_estimates_known_1_2_hz_sine():
    fs = 125.0
    time_s = np.arange(60 * int(fs)) / fs
    references = pd.DataFrame(
        {
            "Time [s]": np.arange(60),
            "HR": 72.0,
            "PULSE": 72.0,
            "RESP": 15.0,
            "SpO2": 97.0,
        }
    )
    waveform = Waveform(
        "sine-check",
        time_s,
        fs,
        {
            "II": np.sin(2 * np.pi * 1.2 * time_s),
            "PLETH": np.sin(2 * np.pi * 1.2 * time_s),
            "RESP": np.sin(2 * np.pi * 0.25 * time_s),
        },
        references,
        "SYNTHETIC",
    )
    result = estimate_ecg_hr(waveform)
    assert np.allclose(result["value"], 72.0, atol=0.1)


def test_ecg_synthetic_rates_match_reference(waveform):
    result = estimate_ecg_hr(waveform)
    reference = waveform.numerics["HR"].mean()
    assert result["quality"].eq("OK").all()
    assert abs(result["value"].mean() - reference) < 0.1


def test_ppg_rate_matches_pulse_reference(waveform):
    result = estimate_ppg_pr(waveform)
    assert result["quality"].eq("OK").all()
    assert abs(result["value"].mean() - waveform.numerics["PULSE"].mean()) < 0.1


def test_ppg_all_missing_samples_get_missing_quality(waveform):
    signals = dict(waveform.signals)
    signals["PLETH"] = np.full_like(signals["PLETH"], np.nan)
    result = estimate_ppg_pr(replace(waveform, signals=signals))
    assert result["quality"].eq("MISSING").all()
    assert result["value"].isna().all()


def test_respiration_rate_matches_reference(waveform):
    result = estimate_resp_rate(waveform)
    assert result["quality"].eq("OK").all()
    assert abs(result["value"].mean() - waveform.numerics["RESP"].mean()) < 0.1


def test_respiration_all_missing_samples_get_missing_quality(waveform):
    signals = dict(waveform.signals)
    signals["RESP"] = np.full_like(signals["RESP"], np.nan)
    result = estimate_resp_rate(replace(waveform, signals=signals))
    assert result["quality"].eq("MISSING").all()
