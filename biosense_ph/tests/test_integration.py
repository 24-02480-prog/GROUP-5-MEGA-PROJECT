from __future__ import annotations

import pytest

from src.schema import validate_feature_table
from src.t08_integration import run_pipeline


def test_pipeline_runs_end_to_end_for_one_record(waveform, tmp_path):
    result = run_pipeline(
        record_ids=["01"],
        scenarios=["A"],
        data_source="synthetic",
        output_dir=tmp_path,
    )
    assert result["fused_timeline"]["record_id"].eq(waveform.record_id).all()
    assert result["fused_timeline"]["scenario"].eq("A").all()
    validate_feature_table(result["feature_table"])
    assert (tmp_path / "fused_timeline.csv").is_file()
    assert (tmp_path / "error_table.csv").is_file()
    assert (tmp_path / "ml_rate_correction_metrics.csv").is_file()
    assert (tmp_path / "quality_cv_metrics.csv").is_file()
    assert "HR_PR_diff" in set(result["feature_table"]["feature"])


def test_pipeline_rejects_unknown_scenarios(tmp_path):
    with pytest.raises(ValueError, match="scenarios must contain"):
        run_pipeline(
            record_ids=["01"],
            scenarios=["D"],
            data_source="synthetic",
            output_dir=tmp_path,
        )
