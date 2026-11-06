"""Shared filtering, windowing, and reference-comparison helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, sosfiltfilt, welch

from .config import get_config
from .schema import FEATURE_COLUMNS, source_labeled_feature_table, validate_feature_table
from .t01_loader import Waveform

WINDOW_SECONDS = 30.0


def windows(waveform: Waveform) -> Iterator[tuple[float, float, slice]]:
    """Yield non-overlapping 30-second signal windows."""
    samples_per_window = int(round(WINDOW_SECONDS * waveform.fs))
    origin = float(waveform.time_s[0])
    total_duration = float(waveform.time_s[-1] - origin + 1.0 / waveform.fs)
    window_count = int(np.ceil(total_duration / WINDOW_SECONDS))
    for window_index in range(window_count):
        begin = window_index * samples_per_window
        finish = min(begin + samples_per_window, len(waveform.time_s))
        if begin >= finish:
            break
        start_s = origin + window_index * WINDOW_SECONDS
        end_s = min(start_s + WINDOW_SECONDS, origin + total_duration)
        yield start_s, end_s, slice(begin, finish)


def filtered_signal(
    values: np.ndarray, fs: float, low_hz: float, high_hz: float
) -> tuple[np.ndarray, float]:
    """Interpolate limited missing samples and apply a zero-phase Butterworth filter."""
    signal = np.asarray(values, dtype=float)
    missing_fraction = float(np.mean(~np.isfinite(signal)))
    if missing_fraction >= 0.2:
        return np.full_like(signal, np.nan), missing_fraction
    valid = np.isfinite(signal)
    if not valid.all():
        indexes = np.arange(signal.size)
        signal = np.interp(indexes, indexes[valid], signal[valid])
    if signal.size < max(30, int(fs * 2)):
        return np.full_like(signal, np.nan), missing_fraction
    sos = butter(
        3, (low_hz, high_hz), btype="bandpass", fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal), missing_fraction


def estimate_rate_diagnostics(
    waveform: Waveform,
    signal_name: str,
) -> list[dict[str, float | str]]:
    """Estimate peak and Welch rates with refractory and median-IBI safeguards."""
    from scipy.signal import find_peaks

    rate_config = get_config()["rates"][{
        "II": "ecg",
        "PLETH": "ppg",
        "RESP": "respiration",
    }[signal_name]]
    filtered, nan_fraction = filtered_signal(
        waveform.signals[signal_name],
        waveform.fs,
        float(rate_config["low_hz"]),
        float(rate_config["high_hz"]),
    )
    if np.isfinite(filtered).any():
        prominence = max(
            float(rate_config["prominence_floor"]),
            float(rate_config["prominence_std_fraction"])
            * float(np.nanstd(filtered)),
        )
        peaks, _ = find_peaks(
            filtered,
            distance=max(
                1,
                int(round(float(rate_config["refractory_seconds"]) * waveform.fs)),
            ),
            prominence=prominence,
        )
    else:
        peaks = np.array([], dtype=int)

    diagnostics = []
    median_kernel = int(rate_config["interval_median_kernel"])
    if median_kernel % 2 == 0:
        raise ValueError("interval_median_kernel must be odd")
    for start_s, end_s, sample_slice in windows(waveform):
        local_peaks = peaks[
            (peaks >= sample_slice.start) & (peaks < sample_slice.stop)
        ]
        if nan_fraction >= 0.2 or local_peaks.size < 2:
            diagnostics.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "peak_rate": np.nan,
                    "ibi_std": np.nan,
                    "spectral_rate": np.nan,
                    "spectral_disagreement": np.nan,
                    "quality": "MISSING",
                }
            )
            continue

        peak_times = waveform.time_s[local_peaks]
        intervals = np.diff(peak_times)
        filtered_intervals = (
            medfilt(intervals, kernel_size=median_kernel)
            if intervals.size >= median_kernel
            else intervals
        )
        interval_center = float(np.median(filtered_intervals))
        interval_outlier_limit = (
            float(rate_config["interval_outlier_fraction"]) * interval_center
        )
        robust_intervals = np.where(
            np.abs(intervals - filtered_intervals) > interval_outlier_limit,
            filtered_intervals,
            intervals,
        )
        peak_rate = float(60.0 * robust_intervals.size / np.sum(robust_intervals))
        ibi_std = float(np.std(robust_intervals))
        local_signal = filtered[sample_slice]
        if not np.isfinite(local_signal).all():
            local_signal = local_signal[np.isfinite(local_signal)]
        if local_signal.size < 2 or not np.isfinite(local_signal).any():
            spectral_rate = np.nan
        else:
            frequencies, power = welch(
                local_signal,
                fs=waveform.fs,
                nperseg=min(
                    local_signal.size,
                    int(
                        float(rate_config["welch_segment_seconds"])
                        * waveform.fs
                    ),
                ),
            )
            band = (
                (frequencies >= float(rate_config["spectral_low_hz"]))
                & (frequencies <= float(rate_config["spectral_high_hz"]))
            )
            spectral_rate = (
                float(frequencies[band][np.argmax(power[band])] * 60.0)
                if band.any()
                else np.nan
            )
            for harmonic in range(
                2, int(rate_config["harmonic_correction_max_divisor"]) + 1
            ):
                subharmonic = spectral_rate / harmonic
                if (
                    subharmonic >= float(rate_config["spectral_low_hz"]) * 60.0
                    and abs(peak_rate - subharmonic) < abs(peak_rate - spectral_rate)
                ):
                    spectral_rate = subharmonic
        disagreement = (
            abs(peak_rate - spectral_rate)
            if np.isfinite(spectral_rate)
            else np.nan
        )
        if signal_name == "PLETH":
            low_amplitude = (
                not np.isfinite(local_signal).any()
                or float(
                    np.nanpercentile(local_signal, 95)
                    - np.nanpercentile(local_signal, 5)
                )
                < 0.08
            )
            variability_threshold = 0.25
        else:
            low_amplitude = False
            variability_threshold = 0.3
        interval_cv = (
            float(np.std(filtered_intervals) / np.mean(filtered_intervals))
            if filtered_intervals.size > 1 and np.mean(filtered_intervals) > 0
            else 0.0
        )
        quality = "OK"
        if (
            np.isfinite(disagreement)
            and disagreement
            > float(rate_config["spectral_disagreement_bpm"])
        ) or interval_cv > variability_threshold or low_amplitude:
            quality = "LOW"
        diagnostics.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "peak_rate": peak_rate,
                "ibi_std": ibi_std,
                "spectral_rate": spectral_rate,
                "spectral_disagreement": disagreement,
                "quality": quality,
            }
        )
    return diagnostics


def build_feature_table(
    waveform: Waveform,
    feature: str,
    unit: str,
    window_results: list[tuple[float, float, float, str]],
) -> pd.DataFrame:
    """Create and validate rows for one estimated rate."""
    frame = pd.DataFrame(
        [
            {
                "record_id": waveform.record_id,
                "start_s": start_s,
                "end_s": end_s,
                "feature": feature,
                "value": value,
                "unit": unit,
                "quality": quality,
            }
            for start_s, end_s, value, quality in window_results
        ],
        columns=FEATURE_COLUMNS,
    )
    frame = source_labeled_feature_table(
        frame, {waveform.record_id: waveform.data_source}
    )
    validate_feature_table(frame)
    return frame


def compare_to_reference(
    waveforms: list[Waveform], feature_tables: list[pd.DataFrame]
) -> pd.DataFrame:
    """Compare estimated rates with matching 1 Hz reference columns."""
    reference_columns = {"HR": "HR", "PR": "PULSE", "RR": "RESP"}
    combined = pd.concat(feature_tables, ignore_index=True)
    rows = []
    waveform_by_id = {}
    for waveform in waveforms:
        waveform_by_id[waveform.record_id] = waveform
        waveform_by_id[
            f"{waveform.data_source.lower()}-{waveform.record_id}"
        ] = waveform
    for (record_id, feature), group in combined.groupby(
        ["record_id", "feature"], sort=True
    ):
        waveform = waveform_by_id[record_id]
        reference_column = reference_columns[feature]
        numerics = waveform.numerics
        reference_times = numerics["Time [s]"].to_numpy(dtype=float)
        errors: list[float] = []
        for row in group.itertuples(index=False):
            in_window = (reference_times >= row.start_s) & (
                reference_times < row.end_s
            )
            references = pd.to_numeric(
                numerics.loc[in_window, reference_column], errors="coerce"
            ).to_numpy(dtype=float)
            if (
                row.quality != "MISSING"
                and np.isfinite(float(row.value))
                and references.size
                and np.isfinite(references).any()
            ):
                reference_value = float(np.nanmean(references))
                errors.append(abs(float(row.value) - reference_value))
        expected_windows = len(group)
        rows.append(
            {
                "record_id": record_id,
                "data_source": waveform.data_source,
                "rate": feature,
                "MAE": float(np.mean(errors)) if errors else np.nan,
                "windows_evaluated": len(errors),
                "valid_window_coverage": (
                    len(errors) / expected_windows if expected_windows else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
