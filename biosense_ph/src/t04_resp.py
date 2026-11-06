"""Respiration filtering, breath detection, and 30-second rate estimates."""

from __future__ import annotations

import pandas as pd

from .rate_utils import build_feature_table, estimate_rate_diagnostics
from .t01_loader import Waveform


def estimate_resp_rate(waveform: Waveform) -> pd.DataFrame:
    """Estimate RR with refractory, median-IBI, and Welch cross-checks."""
    diagnostics = estimate_rate_diagnostics(waveform, "RESP")
    results = [
        (row["start_s"], row["end_s"], row["peak_rate"], row["quality"])
        for row in diagnostics
    ]

    print(f"T04 RESP | source={waveform.data_source} | record={waveform.record_id}")
    table = build_feature_table(waveform, "RR", "breaths/min", results)
    table.attrs["rate_diagnostics"] = diagnostics
    table.attrs["rate_method"] = "peak-based"
    return table
