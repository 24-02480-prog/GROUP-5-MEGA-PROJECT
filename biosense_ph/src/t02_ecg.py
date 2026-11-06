"""ECG filtering, R-peak detection, and 30-second heart-rate estimates."""

from __future__ import annotations

import pandas as pd

from .rate_utils import build_feature_table, estimate_rate_diagnostics
from .t01_loader import Waveform


def estimate_ecg_hr(waveform: Waveform) -> pd.DataFrame:
    """Estimate ECG HR using refractory peak detection and Welch cross-check."""
    diagnostics = estimate_rate_diagnostics(waveform, "II")
    results = [
        (row["start_s"], row["end_s"], row["peak_rate"], row["quality"])
        for row in diagnostics
    ]

    print(f"T02 ECG | source={waveform.data_source} | record={waveform.record_id}")
    table = build_feature_table(waveform, "HR", "bpm", results)
    table.attrs["rate_diagnostics"] = diagnostics
    table.attrs["rate_method"] = "peak-based"
    return table
