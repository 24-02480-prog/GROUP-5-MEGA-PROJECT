"""Record-wise evaluation of rule and ML signal-quality classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .rate_utils import windows
from .config import get_config
from .scenarios import (
    ScenarioRecord,
    apply_quality_example,
    apply_scenario,
    severity_scenarios,
)
from .t05_quality import assess_signal_quality
from .t01_loader import Waveform

FEATURE_NAMES = [
    "std",
    "range",
    "nan_fraction",
    "flatline_ratio",
    "spike_count",
    "kurtosis",
    "skewness",
    "band_power_ratio",
    "dominant_frequency",
    "autocorrelation_peak",
]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _extract_features(values: np.ndarray, fs: float, signal_name: str) -> dict[str, float]:
    metrics = assess_signal_quality(
        ScenarioRecord(
            record_id="feature-window",
            data_source="SYNTHETIC",
            time_s=np.arange(values.size) / fs,
            fs=fs,
            signals={signal_name: values},
            numerics=pd.DataFrame(),
            scenario="A",
            simulated_flag=False,
        ),
        signal_name,
    ).iloc[0]
    finite = np.isfinite(values)
    clean = np.asarray(values, dtype=float)
    if finite.any() and not finite.all():
        indexes = np.arange(clean.size)
        clean = np.interp(indexes, indexes[finite], clean[finite])
    elif not finite.any():
        clean = np.zeros_like(clean)
    std = float(np.std(clean))
    autocorrelation_peak = 0.0
    if clean.size > 2 and std > 0:
        normalized = (clean - np.mean(clean)) / std
        correlation = correlate(normalized, normalized, mode="full", method="fft")
        correlation = correlation[correlation.size // 2 + 1 :]
        autocorrelation_peak = (
            float(np.max(correlation[: max(1, int(fs * 2))]) / clean.size)
            if correlation.size
            else 0.0
        )
    return {
        "std": std,
        "range": float(np.ptp(clean)) if clean.size else 0.0,
        "nan_fraction": float(metrics["nan_fraction"]),
        "flatline_ratio": float(metrics["flatline_ratio"]),
        "spike_count": float(metrics["spike_count"]),
        "kurtosis": float(pd.Series(clean).kurtosis()) if clean.size > 3 else 0.0,
        "skewness": float(pd.Series(clean).skew()) if clean.size > 2 else 0.0,
        "band_power_ratio": float(metrics["band_power_ratio"]),
        "dominant_frequency": float(metrics["dominant_frequency"]),
        "autocorrelation_peak": autocorrelation_peak,
    }


def _training_rows(waveform: Waveform) -> pd.DataFrame:
    rows = []
    scenario_labels = [("A", 0), ("B", 1), ("C", 1)] + [
        (scenario, 1) for scenario in severity_scenarios()
    ]
    for scenario_label, class_label in scenario_labels:
        scenario: ScenarioRecord = apply_scenario(waveform, scenario_label)
        for signal_name in ("II", "PLETH", "RESP"):
            signal = scenario.signals[signal_name]
            quality_rows = assess_signal_quality(scenario, signal_name)
            for window_index, (start_s, end_s, sample_slice) in enumerate(
                windows(waveform)
            ):
                features = _extract_features(
                    signal[sample_slice], waveform.fs, signal_name
                )
                rows.append(
                    {
                        "record_id": waveform.record_id,
                        "data_source": waveform.data_source,
                        "scenario": scenario_label,
                        "severity": scenario_label,
                        "snr_db": (
                            float(scenario_label.removeprefix("noise_snr_").removesuffix("db"))
                            if scenario_label.startswith("noise_snr_")
                            else np.nan
                        ),
                        "signal": signal_name,
                        "start_s": start_s,
                        "end_s": end_s,
                        "window_index": window_index,
                        "label": class_label,
                        "rule_label": int(
                            quality_rows.iloc[window_index]["quality"] != "OK"
                        ),
                        **features,
                    }
                )
    return pd.DataFrame(rows)


def _new_classifier(name: str):
    seed = int(get_config()["ml_quality"]["random_seed"])
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed
            ),
        )
    raise ValueError(f"Unknown classifier {name}")


def evaluate_quality_classifiers(
    waveforms: list[Waveform], output_dir: Path = OUTPUT_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run record holdout and 5-fold train-record CV with severity reports."""
    if len(waveforms) != 2 or waveforms[0].record_id == waveforms[1].record_id:
        raise ValueError("Record-wise evaluation requires two distinct records")
    dataset = pd.concat(
        [_training_rows(waveform) for waveform in waveforms], ignore_index=True
    )
    model_names = ("RandomForest", "LogisticRegression", "RuleBased")
    predictions: dict[str, list[int]] = {name: [] for name in model_names}
    actual: list[int] = []
    metric_rows = []
    severity_metric_rows = []
    severity_confusion_rows = []
    cv_metric_rows = []
    cv_confusion_rows = []
    importance_rows = []
    cv_folds = int(get_config()["ml_quality"]["cv_folds"])
    cv_seed = int(get_config()["ml_quality"]["random_seed"])

    def append_metrics(
        target_rows: list[dict[str, object]],
        context: dict[str, object],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        target_rows.append(
            {
                **context,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "windows_tested": len(y_true),
            }
        )

    def append_confusion(
        target_rows: list[dict[str, object]],
        context: dict[str, object],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        for actual_label, display_actual in enumerate(("OK", "LOW")):
            for predicted_label, display_predicted in enumerate(("OK", "LOW")):
                target_rows.append(
                    {
                        **context,
                        "actual": display_actual,
                        "predicted": display_predicted,
                        "count": int(matrix[actual_label, predicted_label]),
                    }
                )

    for test_waveform in waveforms:
        train = dataset[dataset["record_id"] != test_waveform.record_id]
        test = dataset[dataset["record_id"] == test_waveform.record_id]
        x_train = train[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        actual.extend(y_test.tolist())
        holdout_predictions: dict[str, np.ndarray] = {}
        for model_name in ("RandomForest", "LogisticRegression"):
            model = _new_classifier(model_name)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test).astype(int)
            holdout_predictions[model_name] = y_pred
            predictions[model_name].extend(y_pred.tolist())
            append_metrics(
                metric_rows,
                {
                    "data_source": test_waveform.data_source,
                    "test_record": test_waveform.record_id,
                    "model": model_name,
                    "evaluation": "record_holdout",
                },
                y_test,
                y_pred,
            )
            if model_name == "RandomForest":
                importance_rows.extend(
                    {
                        "data_source": test_waveform.data_source,
                        "test_record": test_waveform.record_id,
                        "feature": feature_name,
                        "importance": float(importance),
                    }
                    for feature_name, importance in zip(
                        FEATURE_NAMES, model.feature_importances_
                    )
                )
        rule_prediction = test["rule_label"].to_numpy(dtype=int)
        holdout_predictions["RuleBased"] = rule_prediction
        predictions["RuleBased"].extend(rule_prediction.tolist())
        append_metrics(
            metric_rows,
            {
                "data_source": test_waveform.data_source,
                "test_record": test_waveform.record_id,
                "model": "RuleBased",
                "evaluation": "record_holdout",
            },
            y_test,
            rule_prediction,
        )

        for model_name in model_names:
            model_prediction = holdout_predictions[model_name]
            for severity, indexes in test.groupby("severity", sort=True).groups.items():
                positions = test.index.get_indexer(indexes)
                severity_y = y_test[positions]
                severity_pred = model_prediction[positions]
                context = {
                    "data_source": test_waveform.data_source,
                    "test_record": test_waveform.record_id,
                    "model": model_name,
                    "severity": severity,
                    "snr_db": (
                        float(test.loc[indexes, "snr_db"].iloc[0])
                        if pd.notna(test.loc[indexes, "snr_db"].iloc[0])
                        else np.nan
                    ),
                    "evaluation": "record_holdout",
                }
                append_metrics(
                    severity_metric_rows, context, severity_y, severity_pred
                )
                append_confusion(
                    severity_confusion_rows, context, severity_y, severity_pred
                )

        stratified = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=cv_seed
        )
        for fold_index, (train_indices, validation_indices) in enumerate(
            stratified.split(x_train, y_train), start=1
        ):
            x_fold = x_train.iloc[train_indices]
            y_fold = y_train[train_indices]
            x_validation = x_train.iloc[validation_indices]
            y_validation = y_train[validation_indices]
            validation_rows = train.iloc[validation_indices]
            fold_predictions: dict[str, np.ndarray] = {}
            for model_name in ("RandomForest", "LogisticRegression"):
                model = _new_classifier(model_name)
                model.fit(x_fold, y_fold)
                fold_predictions[model_name] = model.predict(x_validation).astype(int)
            fold_predictions["RuleBased"] = validation_rows[
                "rule_label"
            ].to_numpy(dtype=int)
            for model_name, fold_prediction in fold_predictions.items():
                fold_context = {
                    "data_source": test_waveform.data_source,
                    "training_record": train["record_id"].iloc[0],
                    "test_record": test_waveform.record_id,
                    "model": model_name,
                    "fold": fold_index,
                    "evaluation": "train_record_5fold_cv",
                }
                append_metrics(cv_metric_rows, fold_context, y_validation, fold_prediction)
                for severity, indexes in validation_rows.groupby(
                    "severity", sort=True
                ).groups.items():
                    positions = validation_rows.index.get_indexer(indexes)
                    severity_y = y_validation[positions]
                    severity_pred = fold_prediction[positions]
                    severity_context = {
                        **fold_context,
                        "severity": severity,
                        "snr_db": (
                            float(validation_rows.loc[indexes, "snr_db"].iloc[0])
                            if pd.notna(
                                validation_rows.loc[indexes, "snr_db"].iloc[0]
                            )
                            else np.nan
                        ),
                    }
                    append_metrics(
                        cv_metric_rows,
                        severity_context,
                        severity_y,
                        severity_pred,
                    )
                    append_confusion(
                        cv_confusion_rows,
                        severity_context,
                        severity_y,
                        severity_pred,
                    )

    metrics = pd.DataFrame(metric_rows)
    confusion_rows = []
    for model_name in model_names:
        matrix = confusion_matrix(actual, predictions[model_name], labels=[0, 1])
        for actual_label, display_actual in enumerate(("OK", "LOW")):
            for predicted_label, display_predicted in enumerate(("OK", "LOW")):
                confusion_rows.append(
                    {
                        "data_source": ",".join(
                            sorted({waveform.data_source for waveform in waveforms})
                        ),
                        "model": model_name,
                        "actual": display_actual,
                        "predicted": display_predicted,
                        "count": int(matrix[actual_label, predicted_label]),
                    }
                )
    confusion = pd.DataFrame(confusion_rows)
    importance = pd.DataFrame(importance_rows)
    severity_metrics = pd.DataFrame(severity_metric_rows)
    severity_confusion = pd.DataFrame(severity_confusion_rows)
    cv_metrics = pd.DataFrame(cv_metric_rows)
    cv_confusion = pd.DataFrame(cv_confusion_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "quality_classifier_metrics.csv", index=False)
    confusion.to_csv(output_dir / "quality_confusion_matrix.csv", index=False)
    importance.to_csv(output_dir / "quality_feature_importance.csv", index=False)
    severity_metrics.to_csv(output_dir / "quality_severity_metrics.csv", index=False)
    severity_confusion.to_csv(
        output_dir / "quality_severity_confusion_matrix.csv", index=False
    )
    cv_metrics.to_csv(output_dir / "quality_cv_metrics.csv", index=False)
    cv_confusion.to_csv(output_dir / "quality_cv_confusion_matrix.csv", index=False)

    source_label = ",".join(sorted({waveform.data_source for waveform in waveforms}))
    fig, axes = plt.subplots(1, len(model_names), figsize=(13, 4))
    for axis, model_name in zip(axes, model_names):
        matrix = confusion_matrix(actual, predictions[model_name], labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name)
        axis.set_xticks([0, 1], labels=["OK", "LOW"])
        axis.set_yticks([0, 1], labels=["OK", "LOW"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.suptitle(f"Quality classifier confusion matrices | source={source_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_confusion_matrix.png", dpi=150)
    plt.close(fig)

    importance_summary = importance.groupby("feature", as_index=False)["importance"].mean()
    importance_summary = importance_summary.sort_values("importance", ascending=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance_summary["feature"], importance_summary["importance"])
    axis.set_xlabel("Mean RandomForest feature importance")
    axis.set_title(f"Quality feature importance | source={source_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_feature_importance.png", dpi=150)
    plt.close(fig)

    snr_metrics = severity_metrics.dropna(subset=["snr_db"])
    fig, axis = plt.subplots(figsize=(7, 5))
    for model_name in model_names:
        selected = snr_metrics[snr_metrics["model"] == model_name]
        summary = selected.groupby("snr_db")["F1"].mean().sort_index(ascending=False)
        if not summary.empty:
            axis.plot(summary.index, summary.values, marker="o", label=model_name)
    axis.set_xlabel("Input SNR (dB)")
    axis.set_ylabel("F1 score")
    axis.set_ylim(0, 1.05)
    axis.set_title(f"Quality classification F1 vs noise level | source={source_label}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_f1_vs_snr.png", dpi=150)
    plt.close(fig)

    print(f"T05 ML QUALITY | source={source_label}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return metrics, confusion, importance


def scenario_quality_table(
    waveform: Waveform, output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """Check clean, noise/spike, and flatline/missing examples per signal."""
    examples = (
        ("A", "clean-control", "OK", "A"),
        ("B", "noise-only", "LOW", "noise-only"),
        ("B", "spikes-only", "LOW", "spikes-only"),
        ("C", "flatline-missing", "MISSING", "flatline-missing"),
    )
    rows = []
    for scenario_label, example_label, expected_quality, example_kind in examples:
        variant = (
            apply_scenario(waveform, scenario_label)
            if example_kind == "A"
            else apply_quality_example(waveform, example_kind)
        )
        for signal_name in ("II", "PLETH", "RESP"):
            per_window = assess_signal_quality(variant, signal_name)
            observed = max(
                per_window["quality"],
                key={"OK": 0, "LOW": 1, "MISSING": 2}.__getitem__,
            )
            rows.append(
                {
                    "record_id": waveform.record_id,
                    "data_source": waveform.data_source,
                    "scenario": scenario_label,
                    "simulated_flag": variant.simulated_flag,
                    "signal": signal_name,
                    "example": example_label,
                    "expected_quality": expected_quality,
                    "observed_quality": observed,
                    "result": "PASS" if observed == expected_quality else "FAIL",
                }
            )
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "scenario_quality_pass_fail.csv", index=False)
    print("T05 SCENARIO QUALITY | source=" + waveform.data_source)
    print(result.to_string(index=False))
    return result
