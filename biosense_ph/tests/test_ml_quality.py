from __future__ import annotations

import pandas as pd

from src.t05_ml_quality import evaluate_quality_classifiers, scenario_quality_table


def test_scenario_pass_fail_table_covers_each_signal(waveform, tmp_path):
    result = scenario_quality_table(waveform, tmp_path)
    assert len(result) == 12
    assert set(result["signal"]) == {"II", "PLETH", "RESP"}
    assert set(result["example"]) == {
        "clean-control",
        "noise-only",
        "spikes-only",
        "flatline-missing",
    }
    assert result["result"].eq("PASS").all()
    assert result["data_source"].eq("SYNTHETIC").all()


def test_recordwise_classifiers_and_rule_baseline_export(two_waveforms, tmp_path):
    metrics, confusion, importance = evaluate_quality_classifiers(
        two_waveforms, tmp_path
    )
    assert set(metrics["test_record"]) == {"synthetic-01", "synthetic-02"}
    assert set(metrics["model"]) == {
        "RandomForest",
        "LogisticRegression",
        "RuleBased",
    }
    assert len(confusion) == 12
    assert set(importance["feature"]) >= {"std", "range", "nan_fraction"}
    assert (tmp_path / "quality_confusion_matrix.png").is_file()
    severity_metrics = pd.read_csv(tmp_path / "quality_severity_metrics.csv")
    cv_metrics = pd.read_csv(tmp_path / "quality_cv_metrics.csv")
    cv_confusion = pd.read_csv(tmp_path / "quality_cv_confusion_matrix.csv")
    assert {"noise_snr_20db", "noise_snr_10db", "noise_snr_5db", "noise_snr_0db"}.issubset(
        set(severity_metrics["severity"])
    )
    assert {"accuracy", "precision", "recall", "F1"}.issubset(cv_metrics.columns)
    assert cv_metrics.loc[
        cv_metrics["evaluation"] == "train_record_5fold_cv", "fold"
    ].nunique() == 5
    assert not cv_confusion.empty
    assert (tmp_path / "quality_f1_vs_snr.png").is_file()
