"""Record-wise Ridge correction of peak-based physiological rates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import get_config
from .rate_utils import estimate_rate_diagnostics, windows
from .schema import (
    FEATURE_COLUMNS,
    source_record_id,
    validate_feature_table,
)
from .t01_loader import Waveform
from .t02_ecg import estimate_ecg_hr
from .t03_ppg import estimate_ppg_pr
from .t04_resp import estimate_resp_rate
from .t05_quality import window_quality

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
RATE_SPECS = {
    "HR": ("HR", "II", estimate_ecg_hr),
    "PR": ("PULSE", "PLETH", estimate_ppg_pr),
    "RR": ("RESP", "RESP", estimate_resp_rate),
}
FEATURE_NAMES = [
    "peak_rate",
    "ibi_std",
    "spectral_rate",
    "spectral_disagreement",
    "quality_nan_fraction",
    "quality_flatline_ratio",
    "quality_spike_count",
    "quality_std",
    "quality_band_power_ratio",
    "quality_dominant_frequency",
]


def _reference(waveform: Waveform, column: str, start_s: float, end_s: float) -> float:
    times = waveform.numerics["Time [s]"].to_numpy(dtype=float)
    selected = (times >= start_s) & (times < end_s)
    values = pd.to_numeric(
        waveform.numerics.loc[selected, column], errors="coerce"
    ).to_numpy(dtype=float)
    return (
        float(np.nanmean(values))
        if values.size and np.isfinite(values).any()
        else np.nan
    )


def _record_windows(waveform: Waveform, feature: str) -> pd.DataFrame:
    reference_column, signal_name, estimator = RATE_SPECS[feature]
    estimates = estimator(waveform)
    diagnostics = estimate_rate_diagnostics(waveform, signal_name)
    diagnostic_by_start = {float(row["start_s"]): row for row in diagnostics}
    rows = []
    for start_s, end_s, sample_slice in windows(waveform):
        diagnostic = diagnostic_by_start[start_s]
        quality = window_quality(
            waveform.signals[signal_name][sample_slice], waveform.fs, signal_name
        )
        estimate_row = estimates.loc[estimates["start_s"] == start_s].iloc[0]
        rows.append(
            {
                "record_id": waveform.record_id,
                "data_source": waveform.data_source,
                "start_s": start_s,
                "end_s": end_s,
                "feature": feature,
                "unit": "breaths/min" if feature == "RR" else "bpm",
                "peak_rate": float(estimate_row["value"])
                if pd.notna(estimate_row["value"])
                else np.nan,
                "rate_quality": str(estimate_row["quality"]),
                "reference": _reference(waveform, reference_column, start_s, end_s),
                "ibi_std": float(diagnostic["ibi_std"]),
                "spectral_rate": float(diagnostic["spectral_rate"]),
                "spectral_disagreement": float(
                    diagnostic["spectral_disagreement"]
                ),
                "quality_nan_fraction": float(quality["nan_fraction"]),
                "quality_flatline_ratio": float(quality["flatline_ratio"]),
                "quality_spike_count": float(quality["spike_count"]),
                "quality_std": float(quality["std"]),
                "quality_band_power_ratio": float(quality["band_power_ratio"]),
                "quality_dominant_frequency": float(
                    quality["dominant_frequency"]
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_rate_correction(
    waveforms: list[Waveform], output_dir: Path = OUTPUT_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on one record and test on the other without test-record tuning."""
    if len(waveforms) != 2 or waveforms[0].record_id == waveforms[1].record_id:
        raise ValueError("Rate correction requires two distinct records")
    record_features = {
        waveform.record_id: {
            feature: _record_windows(waveform, feature)
            for feature in RATE_SPECS
        }
        for waveform in waveforms
    }
    prediction_rows = []
    metric_rows = []
    corrected_feature_rows = []
    peak_feature_rows = []
    alpha = float(get_config()["ml_rate"]["ridge_alpha"])

    for training_waveform, test_waveform in (
        (waveforms[0], waveforms[1]),
        (waveforms[1], waveforms[0]),
    ):
        for feature in RATE_SPECS:
            train = record_features[training_waveform.record_id][feature]
            test = record_features[test_waveform.record_id][feature]
            valid_train = (
                np.isfinite(train["peak_rate"])
                & np.isfinite(train["reference"])
                & train[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            )
            train = train.loc[valid_train]
            valid_test = np.isfinite(test["peak_rate"]) & np.isfinite(test["reference"])
            if len(train) < 3 or not valid_test.any():
                raise ValueError(
                    f"Insufficient valid windows for ML correction of {feature}"
                )
            x_train = train[FEATURE_NAMES].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            target_correction = train["reference"] - train["peak_rate"]
            model = Ridge(alpha=alpha)
            model.fit(x_train, target_correction)

            x_test = test[FEATURE_NAMES].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            corrected = test["peak_rate"].to_numpy(dtype=float).copy()
            corrected_mask = np.isfinite(corrected)
            corrected[corrected_mask] += model.predict(x_test.loc[corrected_mask])
            evaluation_mask = valid_test.to_numpy()
            peak_errors = np.abs(
                test.loc[valid_test, "peak_rate"].to_numpy(dtype=float)
                - test.loc[valid_test, "reference"].to_numpy(dtype=float)
            )
            corrected_errors = np.abs(
                corrected[evaluation_mask]
                - test.loc[valid_test, "reference"].to_numpy(dtype=float)
            )
            metric_rows.append(
                {
                    "data_source": test_waveform.data_source,
                    "training_record": training_waveform.record_id,
                    "test_record": test_waveform.record_id,
                    "rate": feature,
                    "peak_based_MAE": float(np.mean(peak_errors)),
                    "ml_corrected_MAE": float(np.mean(corrected_errors)),
                    "peak_based_method": "peak-based (median-filtered IBI)",
                    "corrected_method": "ML-corrected (Ridge)",
                    "windows_evaluated": int(np.sum(evaluation_mask)),
                }
            )

            for position, row in enumerate(test.itertuples(index=False)):
                peak_value = float(row.peak_rate) if np.isfinite(row.peak_rate) else np.nan
                corrected_value = corrected[position]
                quality = row.rate_quality
                if quality == "MISSING":
                    peak_value = corrected_value = np.nan
                record_id = source_record_id(row.record_id, row.data_source)
                common = {
                    "record_id": record_id,
                    "start_s": row.start_s,
                    "end_s": row.end_s,
                    "feature": feature,
                    "unit": row.unit,
                    "quality": quality,
                }
                peak_feature_rows.append({**common, "value": peak_value})
                corrected_feature_rows.append(
                    {**common, "value": corrected_value}
                )
                prediction_rows.append(
                    {
                        "record_id": record_id,
                        "data_source": row.data_source,
                        "start_s": row.start_s,
                        "end_s": row.end_s,
                        "rate": feature,
                        "peak_based_estimate": peak_value,
                        "ml_corrected_estimate": corrected_value,
                        "provided_reference": row.reference,
                        "quality": quality,
                        "peak_based_method": "peak-based (median-filtered IBI)",
                        "corrected_method": "ML-corrected (Ridge)",
                        "training_record": training_waveform.record_id,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    corrected_features = pd.DataFrame(
        corrected_feature_rows, columns=FEATURE_COLUMNS
    )
    peak_features = pd.DataFrame(peak_feature_rows, columns=FEATURE_COLUMNS)
    validate_feature_table(corrected_features)
    validate_feature_table(peak_features)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "ml_rate_correction_metrics.csv", index=False)
    predictions.to_csv(output_dir / "ml_rate_correction_predictions.csv", index=False)
    corrected_features.to_csv(
        output_dir / "ml_corrected_feature_table.csv", index=False
    )
    peak_features.to_csv(output_dir / "peak_based_feature_table.csv", index=False)
    return metrics, predictions, corrected_features, peak_features
