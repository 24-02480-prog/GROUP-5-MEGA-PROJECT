"""Shared feature-table schema and validation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "record_id",
    "start_s",
    "end_s",
    "feature",
    "value",
    "unit",
    "quality",
]

FEATURE_UNITS: Mapping[str, str] = {
    "HR": "bpm",
    "PR": "bpm",
    "HR_PR_diff": "bpm",
    "RR": "breaths/min",
    "SpO2_ref": "%",
    "quality": "label",
    "scenario_flag": "label",
}
QUALITY_LABELS = {"OK", "LOW", "MISSING"}


def validate_feature_table(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a feature table violates the shared schema."""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    extra = [column for column in df.columns if column not in FEATURE_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(map(str, extra))}")
        raise ValueError("Invalid feature table columns (" + "; ".join(details) + ")")

    if df.empty:
        return

    if df["record_id"].isna().any() or df["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("record_id must contain a non-empty value in every row")

    for column in ("start_s", "end_s"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values")
        df_values = values.to_numpy(dtype=float)
        if column == "start_s":
            starts = df_values
        else:
            ends = df_values
    if np.any(ends <= starts):
        raise ValueError("Each end_s must be greater than its start_s")

    unknown_features = sorted(set(df["feature"].dropna()) - set(FEATURE_UNITS))
    if unknown_features:
        raise ValueError(f"Unknown feature value(s): {', '.join(map(str, unknown_features))}")
    if df["feature"].isna().any():
        raise ValueError("feature must not be missing")

    expected_units = df["feature"].map(FEATURE_UNITS)
    wrong_units = df["unit"].astype(str) != expected_units
    if wrong_units.any():
        rows = df.index[wrong_units].tolist()
        raise ValueError(f"unit does not match feature in row(s): {rows}")

    invalid_quality = df["quality"].isna() | ~df["quality"].isin(QUALITY_LABELS)
    if invalid_quality.any():
        rows = df.index[invalid_quality].tolist()
        raise ValueError(
            "quality must be exactly one of OK, LOW, MISSING; "
            f"invalid row(s): {rows}"
        )

    label_features = df["feature"].isin({"quality", "scenario_flag"})
    numeric_features = ~label_features
    numeric_values = pd.to_numeric(df.loc[numeric_features, "value"], errors="coerce")
    invalid_numeric = numeric_values.isna() | ~np.isfinite(
        numeric_values.to_numpy(dtype=float)
    )
    invalid_numeric &= df.loc[numeric_features, "quality"] == "OK"
    if invalid_numeric.any():
        rows = numeric_values.index[invalid_numeric].tolist()
        raise ValueError(
            "Numeric feature value must be finite when quality is OK; "
            f"invalid row(s): {rows}"
        )
    invalid_labels = df.loc[label_features, "value"].isna()
    if invalid_labels.any():
        rows = invalid_labels.index[invalid_labels].tolist()
        raise ValueError(f"Label feature value must not be missing in row(s): {rows}")

    intervals = df.assign(
        _start=pd.to_numeric(df["start_s"]),
        _end=pd.to_numeric(df["end_s"]),
    ).sort_values(["record_id", "feature", "_start", "_end"])
    for (_, feature), group in intervals.groupby(["record_id", "feature"], sort=False):
        prior_end = group["_end"].shift()
        overlaps = group["_start"] < prior_end
        if overlaps.fillna(False).any():
            raise ValueError(
                f"Windows for feature {feature!r} overlap within a record"
            )


def source_record_id(record_id: str, data_source: str) -> str:
    """Return a record ID that carries its source without adding a schema column."""
    prefix = f"{data_source.lower()}-"
    return record_id if record_id.lower().startswith(prefix) else prefix + record_id


def source_labeled_feature_table(
    df: pd.DataFrame, source_by_record: Mapping[str, str]
) -> pd.DataFrame:
    """Keep the shared schema while making source visible in each record ID."""
    result = df.copy()
    source_by_labeled_id = {
        source_record_id(record_id, source): source
        for record_id, source in source_by_record.items()
    }
    labeled_ids = []
    for record_id in result["record_id"].astype(str):
        source = source_by_record.get(
            record_id, source_by_labeled_id.get(record_id)
        )
        if source is None:
            raise ValueError(f"Missing data source for record_id value: {record_id}")
        labeled_ids.append(source_record_id(record_id, source))
    result["record_id"] = labeled_ids
    validate_feature_table(result)
    return result
