from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema import (
    FEATURE_COLUMNS,
    source_labeled_feature_table,
    validate_feature_table,
)


def _feature_table(value=72.0, quality="OK"):
    return pd.DataFrame(
        [["synthetic-01", 0.0, 30.0, "HR", value, "bpm", quality]],
        columns=FEATURE_COLUMNS,
    )


def test_schema_accepts_expected_rate_row():
    validate_feature_table(_feature_table())


def test_schema_rejects_bad_quality_label():
    with pytest.raises(ValueError, match="quality must be exactly"):
        validate_feature_table(_feature_table(quality="BAD"))


def test_schema_accepts_gated_missing_value_and_rejects_overlap():
    missing = _feature_table(value=np.nan, quality="MISSING")
    validate_feature_table(missing)
    overlapping = pd.concat(
        [missing, missing.assign(start_s=20.0, end_s=50.0)], ignore_index=True
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_feature_table(overlapping)


def test_source_is_visible_in_feature_record_id_without_extra_column():
    real_row = _feature_table().assign(record_id="01")
    labeled = source_labeled_feature_table(real_row, {"01": "REAL"})
    assert labeled.loc[0, "record_id"] == "real-01"
    assert labeled.columns.tolist() == FEATURE_COLUMNS
