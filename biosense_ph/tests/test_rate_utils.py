from __future__ import annotations

import numpy as np

from src.rate_utils import filtered_signal, windows


def test_windows_are_non_overlapping_thirty_second_windows(waveform):
    result = list(windows(waveform))
    assert len(result) == 16
    assert result[0][0:2] == (0.0, 30.0)
    assert result[0][2].stop == result[1][2].start


def test_filter_preserves_shape_and_handles_missing_samples():
    fs = 125.0
    time_s = np.arange(30 * int(fs)) / fs
    signal = np.sin(2 * np.pi * 1.2 * time_s)
    signal[100] = np.nan
    filtered, nan_fraction = filtered_signal(signal, fs, 0.5, 8.0)
    assert filtered.shape == signal.shape
    assert nan_fraction > 0
    assert np.isfinite(filtered).all()


def test_filter_marks_heavily_missing_window_unavailable():
    signal = np.full(30 * 125, np.nan)
    filtered, fraction = filtered_signal(signal, 125.0, 0.5, 8.0)
    assert fraction == 1.0
    assert np.isnan(filtered).all()
