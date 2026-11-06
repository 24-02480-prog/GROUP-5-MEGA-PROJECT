"""Scenario-wise rate evaluation and source-labeled figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .rate_utils import filtered_signal
from .scenarios import ScenarioRecord, apply_scenario
from .t02_ecg import estimate_ecg_hr
from .t03_ppg import estimate_ppg_pr
from .t04_resp import estimate_resp_rate
from .t05_quality import assess_signal_quality
from .t06_fusion import fuse_windows
from .t01_loader import Waveform
from .schema import source_labeled_feature_table

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
RATE_CONFIG = {
    "HR": ("HR", "II", estimate_ecg_hr),
    "PR": ("PULSE", "PLETH", estimate_ppg_pr),
    "RR": ("RESP", "RESP", estimate_resp_rate),
}


def _scenario_quality(
    variant: ScenarioRecord,
) -> dict[str, pd.DataFrame]:
    return {
        signal_name: assess_signal_quality(variant, signal_name)
        for signal_name in ("II", "PLETH", "RESP")
    }


def _reference_window(
    waveform: Waveform, column: str, start_s: float, end_s: float
) -> float:
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


def rate_error_row(
    waveform: Waveform,
    variant: ScenarioRecord,
    feature: str,
    reference_column: str,
    signal_name: str,
    estimate: pd.DataFrame,
    signal_quality: pd.DataFrame,
) -> dict[str, object]:
    quality_by_start = {
        float(row.start_s): str(row.quality)
        for row in signal_quality.itertuples(index=False)
    }
    errors_before = []
    errors_after = []
    window_count = 0
    for row in estimate.itertuples(index=False):
        window_count += 1
        reference = _reference_window(
            waveform, reference_column, float(row.start_s), float(row.end_s)
        )
        value = float(row.value) if pd.notna(row.value) else np.nan
        if np.isfinite(reference) and np.isfinite(value):
            errors_before.append(abs(value - reference))
            if row.quality == "OK" and quality_by_start.get(float(row.start_s)) == "OK":
                errors_after.append(abs(value - reference))
    return {
        "record_id": waveform.record_id,
        "data_source": waveform.data_source,
        "scenario": variant.scenario,
        "rate": feature,
        "signal": signal_name,
        "MAE_before": float(np.mean(errors_before)) if errors_before else np.nan,
        "windows_evaluated_before": len(errors_before),
        "valid_window_coverage_before": (
            len(errors_before) / window_count if window_count else 0.0
        ),
        "MAE_after": float(np.mean(errors_after)) if errors_after else np.nan,
        "windows_evaluated_after": len(errors_after),
        "valid_window_coverage_after": (
            len(errors_after) / window_count if window_count else 0.0
        ),
    }


def evaluate_scenarios(
    waveforms: list[Waveform],
    output_dir: Path = OUTPUT_DIR,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate HR/PR/RR before and after gating for selected scenarios."""
    rows = []
    baseline_features = []
    flag_summaries = []
    validation_observations = []
    scenario_list = (
        ["A", "B", "C"]
        if scenarios is None
        else scenarios
    )
    for waveform in waveforms:
        for scenario_label in scenario_list:
            variant = apply_scenario(waveform, scenario_label)
            quality = _scenario_quality(variant)
            rate_tables = []
            for feature, (reference_column, signal_name, estimator) in RATE_CONFIG.items():
                estimate = estimator(variant)
                rate_tables.append(estimate)
                signal_quality = quality[signal_name]
                quality_by_start = {
                    float(row.start_s): str(row.quality)
                    for row in signal_quality.itertuples(index=False)
                }
                for estimate_row in estimate.itertuples(index=False):
                    reference = _reference_window(
                        waveform,
                        reference_column,
                        float(estimate_row.start_s),
                        float(estimate_row.end_s),
                    )
                    estimate_value = (
                        float(estimate_row.value)
                        if pd.notna(estimate_row.value)
                        else np.nan
                    )
                    row_quality = max(
                        (
                            str(estimate_row.quality),
                            quality_by_start.get(float(estimate_row.start_s), "OK"),
                        ),
                        key={"OK": 0, "LOW": 1, "MISSING": 2}.__getitem__,
                    )
                    validation_observations.append(
                        {
                            "record_id": waveform.record_id,
                            "data_source": waveform.data_source,
                            "scenario": scenario_label,
                            "rate": feature,
                            "start_s": estimate_row.start_s,
                            "reference": reference,
                            "estimate": estimate_value,
                            "absolute_error": (
                                abs(estimate_value - reference)
                                if np.isfinite(estimate_value)
                                and np.isfinite(reference)
                                else np.nan
                            ),
                            "quality": row_quality,
                        }
                    )
                rows.append(
                    rate_error_row(
                        waveform,
                        variant,
                        feature,
                        reference_column,
                        signal_name,
                        estimate,
                        quality[signal_name],
                    )
                )
            timeline, feature_table = fuse_windows(
                variant,
                rate_tables,
                quality,
                scenario=scenario_label,
                simulated_flag=variant.simulated_flag,
            )
            flag_summary = (
                timeline.groupby(
                    ["record_id", "data_source", "scenario", "demo_flag"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "windows"})
            )
            flag_summaries.append(flag_summary)
            if scenario_label == "A":
                baseline_features.append(feature_table)

    errors = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_dir / "error_table.csv", index=False)
    feature_table = source_labeled_feature_table(
        pd.concat(baseline_features, ignore_index=True),
        {waveform.record_id: waveform.data_source for waveform in waveforms},
    )
    feature_table.to_csv(output_dir / "final_feature_table.csv", index=False)
    pd.concat(flag_summaries, ignore_index=True).to_csv(
        output_dir / "scenario_flag_summary.csv", index=False
    )
    observations = pd.DataFrame(validation_observations)
    observations.to_csv(output_dir / "validation_observations.csv", index=False)
    _plot_error_summary(errors, output_dir)
    _plot_validation_figures(observations, errors, output_dir)
    _plot_physiological_signals(waveforms[0], output_dir)
    return errors


def _plot_error_summary(errors: pd.DataFrame, output_dir: Path) -> None:
    source_label = ",".join(sorted(errors["data_source"].unique()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for axis, feature in zip(axes, ("HR", "PR", "RR")):
        selected = errors[errors["rate"] == feature]
        summary = selected[selected["scenario"].isin(["A", "B", "C"])].groupby("scenario")[
            ["MAE_before", "MAE_after"]
        ].mean().reindex(["A", "B", "C"])
        x = np.arange(len(summary))
        axis.bar(x - 0.18, summary["MAE_before"], 0.36, label="Before gating")
        axis.bar(x + 0.18, summary["MAE_after"], 0.36, label="After gating")
        axis.set_xticks(x, summary.index)
        axis.set_title(feature)
        axis.set_xlabel("Scenario")
        axis.set_ylabel("MAE (bpm)" if feature in ("HR", "PR") else "MAE (breaths/min)")
        axis.legend(fontsize=8)
    fig.suptitle(f"Rate error before/after quality gating | source={source_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_error_summary.png", dpi=150)
    plt.close(fig)


def _plot_validation_figures(
    observations: pd.DataFrame,
    errors: pd.DataFrame,
    output_dir: Path,
) -> None:
    source_label = ",".join(sorted(observations["data_source"].unique()))
    units = {"HR": "bpm", "PR": "bpm", "RR": "breaths/min"}
    for feature in ("HR", "PR", "RR"):
        selected = observations[
            (observations["rate"] == feature)
            & np.isfinite(observations["estimate"])
            & np.isfinite(observations["reference"])
        ].copy()
        mean_rate = (selected["estimate"] + selected["reference"]) / 2.0
        difference = selected["estimate"] - selected["reference"]
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(mean_rate, difference, alpha=0.65)
        if not difference.empty:
            bias = float(difference.mean())
            spread = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
            axis.axhline(bias, color="#c62828", label=f"Bias = {bias:.2f} {units[feature]}")
            axis.axhline(
                bias + 1.96 * spread,
                color="#555555",
                linestyle="--",
                label="95% agreement limits",
            )
            axis.axhline(
                bias - 1.96 * spread,
                color="#555555",
                linestyle="--",
            )
        axis.set_xlabel(f"Mean estimated/reference {feature} ({units[feature]})")
        axis.set_ylabel(f"Estimate - reference ({units[feature]})")
        axis.set_title(f"Bland-Altman: {feature} | source={source_label}")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"bland_altman_{feature}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for axis, feature in zip(axes, ("HR", "PR", "RR")):
        selected = observations[
            (observations["rate"] == feature)
            & np.isfinite(observations["absolute_error"])
        ]
        labels = ["OK", "LOW", "MISSING"]
        distributions = [
            selected.loc[selected["quality"] == label, "absolute_error"].to_numpy()
            for label in labels
        ]
        nonempty = [(label, values) for label, values in zip(labels, distributions) if values.size]
        if nonempty:
            axis.boxplot(
                [values for _, values in nonempty],
                tick_labels=[label for label, _ in nonempty],
                showfliers=False,
            )
        axis.set_title(feature)
        axis.set_xlabel("Quality label")
        axis.set_ylabel(f"Absolute error ({units[feature]})")
    fig.suptitle(f"Rate error by quality label | source={source_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_quality_label.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, feature in zip(axes, ("HR", "PR", "RR")):
        selected = errors[errors["rate"] == feature]
        for prefix, label, marker in (
            ("before", "Gating off", "o"),
            ("after", "Gating on", "s"),
        ):
            evaluated = selected[f"windows_evaluated_{prefix}"].sum()
            total = len(selected) * 16
            coverage = evaluated / total if total else 0.0
            mae = selected[f"MAE_{prefix}"].mean()
            if np.isfinite(mae):
                axis.scatter(coverage, mae, label=label, marker=marker, s=60)
                axis.annotate(label, (coverage, mae), xytext=(5, 5), textcoords="offset points")
        axis.set_xlim(-0.02, 1.02)
        axis.set_xlabel("Valid-window coverage")
        axis.set_ylabel(f"MAE ({units[feature]})")
        axis.set_title(feature)
        axis.legend(fontsize=8)
    fig.suptitle(f"Coverage-MAE trade-off | source={source_label}")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_mae.png", dpi=150)
    plt.close(fig)


def _peak_parameters(signal_name: str, filtered: np.ndarray, fs: float):
    spread = float(np.nanstd(filtered))
    if signal_name == "II":
        return int(0.45 * fs), max(0.10, 0.35 * spread)
    if signal_name == "PLETH":
        return int(0.45 * fs), max(0.04, 0.12 * spread)
    return int(1.5 * fs), max(0.05, 0.2 * spread)


def _plot_physiological_signals(waveform: Waveform, output_dir: Path) -> None:
    display_info = {
        "II": ("ECG", "mV", 0.5, 40.0),
        "PLETH": ("PPG", "a.u.", 0.5, 8.0),
        "RESP": ("Respiration", "a.u.", 0.1, 0.7),
    }
    for signal_name, (display_name, unit, low_hz, high_hz) in display_info.items():
        raw = waveform.signals[signal_name]
        filtered, _ = filtered_signal(raw, waveform.fs, low_hz, high_hz)
        plot_length = min(len(raw), int(30 * waveform.fs))
        time_s = waveform.time_s[:plot_length]
        raw_plot = raw[:plot_length]
        filtered_plot = filtered[:plot_length]
        if np.isfinite(filtered_plot).any():
            distance, prominence = _peak_parameters(
                signal_name, filtered_plot, waveform.fs
            )
            peak_indexes, _ = find_peaks(
                filtered_plot, distance=distance, prominence=prominence
            )
        else:
            peak_indexes = np.array([], dtype=int)

        quality = assess_signal_quality(
            apply_scenario(waveform, "A"), signal_name
        )
        fig, axis = plt.subplots(figsize=(11, 4))
        for row in quality.itertuples(index=False):
            left = max(0.0, float(row.start_s))
            right = min(30.0, float(row.end_s))
            if left < right:
                color = {
                    "OK": "#d9f2d9",
                    "LOW": "#ffe6b3",
                    "MISSING": "#dddddd",
                }[row.quality]
                axis.axvspan(left, right, color=color, alpha=0.25, zorder=0)
        axis.plot(time_s, raw_plot, color="0.65", linewidth=0.7, label="Raw signal")
        axis.plot(
            time_s,
            filtered_plot,
            color="#1769aa",
            linewidth=0.9,
            label="Band-pass processed",
        )
        if peak_indexes.size:
            axis.scatter(
                time_s[peak_indexes],
                filtered_plot[peak_indexes],
                s=18,
                color="#c62828",
                label="Detected peaks",
                zorder=3,
            )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"Amplitude ({unit})")
        axis.set_title(
            f"{display_name} raw/processed signal and peaks | "
            f"source={waveform.data_source} | record={waveform.record_id} | scenario=A"
        )
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{display_name.lower()}_signal.png", dpi=150)
        plt.close(fig)
