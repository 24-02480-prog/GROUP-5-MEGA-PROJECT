"""PPG filtering, pulse-peak detection, and 30-second pulse-rate estimates."""

from __future__ import annotations

import pandas as pd

from .rate_utils import build_feature_table, estimate_rate_diagnostics
from .t01_loader import Waveform


def estimate_ppg_pr(waveform: Waveform) -> pd.DataFrame:
    """Estimate PPG PR with refractory, median-IBI, and Welch cross-checks."""
    diagnostics = estimate_rate_diagnostics(waveform, "PLETH")
    results = [
        (row["start_s"], row["end_s"], row["peak_rate"], row["quality"])
        for row in diagnostics
    ]

    print(f"T03 PPG | source={waveform.data_source} | record={waveform.record_id}")
    table = build_feature_table(waveform, "PR", "bpm", results)
    table.attrs["rate_diagnostics"] = diagnostics
    table.attrs["rate_method"] = "peak-based"
    return table
