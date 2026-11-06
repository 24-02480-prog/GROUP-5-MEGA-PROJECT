"""Rule-based signal quality checks and window-level quality features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from .config import get_config
from .rate_utils import windows
from .scenarios import ScenarioRecord
from .schema import (
    FEATURE_COLUMNS,
    source_labeled_feature_table,
    validate_feature_table,
)
from .t01_loader import Waveform

LOW_FLATLINE_RATIO = 0.20
LOW_SPIKE_COUNT = 250
LOW_BAND_POWER_RATIO = {
    "II": 0.75,
    "PLETH": 0.15,
    "RESP": 0.15,
    "V": 0.75,
    "AVR": 0.75,
}
LOW_STD = 0.001
EXTREME_ROBUST_Z = 80.0
FLATLINE_EPS = 1e-8

TARGET_BANDS = {
    "II": (0.5, 40.0),
    "PLETH": (0.5, 8.0),
    "RESP": (0.1, 0.7),
    "V": (0.5, 40.0),
    "AVR": (0.5, 40.0),
}


def window_quality(values: np.ndarray, fs: float, signal_name: str) -> dict[str, float | str]:
    """Calculate editable rule features and assign OK, LOW, or MISSING."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    nan_fraction = float(1 - finite.mean()) if values.size else 1.0
    quality_config = get_config()["quality"]
    if nan_fraction >= float(quality_config["missing_nan_fraction"]) or not finite.any():
        return {
            "quality": "MISSING",
            "nan_fraction": nan_fraction,
            "flatline_ratio": 0.0,
            "spike_count": 0.0,
            "std": np.nan,
            "band_power_ratio": 0.0,
            "dominant_frequency": np.nan,
        }

    indexes = np.arange(values.size)
    clean = np.interp(indexes, indexes[finite], values[finite])
    differences = np.diff(clean)
    flatline_ratio = (
        float(np.mean(np.abs(differences) <= FLATLINE_EPS))
        if differences.size
        else 1.0
    )
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    robust_scale = max(1.4826 * mad, np.finfo(float).eps)
    robust_z = np.abs(clean - center) / robust_scale
    spike_count = float(np.sum(robust_z > 8.0))
    std = float(np.std(clean))

    frequencies, power = periodogram(clean, fs=fs)
    total_power = float(np.sum(power[1:]))
    low_hz, high_hz = TARGET_BANDS.get(signal_name, (0.1, min(40.0, fs / 2)))
    in_band = (frequencies >= low_hz) & (frequencies <= min(high_hz, fs / 2))
    band_power_ratio = (
        float(np.sum(power[in_band]) / total_power) if total_power > 0 else 0.0
    )
    dominant_frequency = (
        float(frequencies[1:][np.argmax(power[1:])])
        if frequencies.size > 1 and np.any(power[1:])
        else 0.0
    )

    low_quality = (
        nan_fraction >= float(quality_config["partial_dropout_low_fraction"])
        or
        flatline_ratio >= LOW_FLATLINE_RATIO
        or spike_count >= LOW_SPIKE_COUNT
        or band_power_ratio
        < LOW_BAND_POWER_RATIO.get(signal_name, 0.15)
        or std < LOW_STD
        or float(np.max(robust_z)) > EXTREME_ROBUST_Z
    )
    return {
        "quality": "LOW" if low_quality else "OK",
        "nan_fraction": nan_fraction,
        "flatline_ratio": flatline_ratio,
        "spike_count": spike_count,
        "std": std,
        "band_power_ratio": band_power_ratio,
        "dominant_frequency": dominant_frequency,
    }


def assess_signal_quality(
    waveform: Waveform | ScenarioRecord, signal_name: str
) -> pd.DataFrame:
    """Return per-window signal metrics for quality reports and scenario checks."""
    if signal_name not in waveform.signals:
        raise ValueError(f"Signal {signal_name!r} is not available")
    rows = []
    for start_s, end_s, sample_slice in windows(waveform):
        metrics = window_quality(
            waveform.signals[signal_name][sample_slice], waveform.fs, signal_name
        )
        rows.append(
            {
                "record_id": waveform.record_id,
                "data_source": waveform.data_source,
                "scenario": getattr(waveform, "scenario", "A"),
                "simulated_flag": getattr(waveform, "simulated_flag", False),
                "signal": signal_name,
                "start_s": start_s,
                "end_s": end_s,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def assess_quality(waveform: Waveform | ScenarioRecord) -> pd.DataFrame:
    """Return one schema-validated worst-case quality label per time window."""
    signal_tables = [
        assess_signal_quality(waveform, name)
        for name in ("II", "PLETH", "RESP")
        if name in waveform.signals
    ]
    if not signal_tables:
        raise ValueError("No supported ECG, PPG, or respiration signal is available")
    details = pd.concat(signal_tables, ignore_index=True)
    priority = {"OK": 0, "LOW": 1, "MISSING": 2}
    rows = []
    for (start_s, end_s), group in details.groupby(["start_s", "end_s"], sort=True):
        label = max(group["quality"], key=priority.__getitem__)
        rows.append(
            {
                "record_id": waveform.record_id,
                "start_s": start_s,
                "end_s": end_s,
                "feature": "quality",
                "value": label,
                "unit": "label",
                "quality": label,
            }
        )
    result = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    result = source_labeled_feature_table(
        result, {waveform.record_id: waveform.data_source}
    )
    validate_feature_table(result)
    print(
        f"T05 RULE QUALITY | source={waveform.data_source} | "
        f"record={waveform.record_id} | scenario={getattr(waveform, 'scenario', 'A')}"
    )
    return result
