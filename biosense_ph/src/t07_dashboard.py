"""Single-page Streamlit dashboard with a matplotlib command-line fallback."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import importlib.util
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate import RATE_CONFIG, rate_error_row
    from src.rate_utils import filtered_signal
    from src.scenarios import ScenarioRecord, apply_scenario
    from src.t01_loader import Waveform, load_records
    from src.t02_ecg import estimate_ecg_hr
    from src.t03_ppg import estimate_ppg_pr
    from src.t04_resp import estimate_resp_rate
    from src.t05_quality import assess_signal_quality
    from src.t06_fusion import fuse_windows
    from src.schema import source_labeled_feature_table
else:
    from .evaluate import RATE_CONFIG, rate_error_row
    from .rate_utils import filtered_signal
    from .scenarios import ScenarioRecord, apply_scenario
    from .t01_loader import Waveform, load_records
    from .t02_ecg import estimate_ecg_hr
    from .t03_ppg import estimate_ppg_pr
    from .t04_resp import estimate_resp_rate
    from .t05_quality import assess_signal_quality
    from .t06_fusion import fuse_windows
    from .schema import source_labeled_feature_table

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
RATE_ESTIMATORS = (estimate_ecg_hr, estimate_ppg_pr, estimate_resp_rate)
SIGNAL_INFO = {
    "II": ("ECG", "mV", 0.5, 40.0, int(0.45 * 125), 0.10, 0.35),
    "PLETH": ("PPG", "a.u.", 0.5, 8.0, int(0.45 * 125), 0.04, 0.12),
    "RESP": ("Respiration", "a.u.", 0.1, 0.7, int(1.5 * 125), 0.05, 0.20),
}


def build_dashboard_data(
    waveform: Waveform, scenario: str
) -> tuple[ScenarioRecord, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the selected scenario's rates, quality, fusion, and metric tables."""
    variant = apply_scenario(waveform, scenario)
    rate_tables = [estimator(variant) for estimator in RATE_ESTIMATORS]
    signal_quality = {
        signal_name: assess_signal_quality(variant, signal_name)
        for signal_name in ("II", "PLETH", "RESP")
    }
    timeline, feature_table = fuse_windows(
        variant,
        rate_tables,
        signal_quality,
        scenario=scenario,
        simulated_flag=variant.simulated_flag,
    )
    feature_table = source_labeled_feature_table(
        feature_table, {waveform.record_id: waveform.data_source}
    )
    metric_rows = []
    for (feature, (reference_column, signal_name, _)), estimate in zip(
        RATE_CONFIG.items(), rate_tables
    ):
        metric_rows.append(
            rate_error_row(
                waveform,
                variant,
                feature,
                reference_column,
                signal_name,
                estimate,
                signal_quality[signal_name],
            )
        )
    return (
        variant,
        signal_quality,
        timeline,
        feature_table,
        pd.DataFrame(metric_rows),
    )


def make_signal_figure(
    variant: ScenarioRecord,
    signal_quality: dict[str, pd.DataFrame],
) -> plt.Figure:
    """Plot raw and processed signals with detected peaks and quality shading."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plot_duration_s = 60.0
    for axis, (signal_name, info) in zip(axes, SIGNAL_INFO.items()):
        display_name, unit, low_hz, high_hz, min_distance, min_prominence, spread_factor = info
        raw = variant.signals[signal_name]
        processed, _ = filtered_signal(raw, variant.fs, low_hz, high_hz)
        count = min(len(raw), int(plot_duration_s * variant.fs))
        time_s = variant.time_s[:count]
        raw_view = raw[:count]
        processed_view = processed[:count]
        quality = signal_quality[signal_name]
        for row in quality.itertuples(index=False):
            left = max(0.0, float(row.start_s))
            right = min(plot_duration_s, float(row.end_s))
            if left < right:
                color = {
                    "OK": "#d9f2d9",
                    "LOW": "#ffe6b3",
                    "MISSING": "#dddddd",
                }[row.quality]
                axis.axvspan(left, right, color=color, alpha=0.25, zorder=0)

        axis.plot(time_s, raw_view, color="0.65", linewidth=0.7, label="Raw signal")
        axis.plot(
            time_s,
            processed_view,
            color="#1769aa",
            linewidth=0.9,
            label="Band-pass processed",
        )
        if np.isfinite(processed_view).any():
            spread = float(np.nanstd(processed_view))
            peaks, _ = find_peaks(
                processed_view,
                distance=max(1, int(min_distance * variant.fs / 125)),
                prominence=max(min_prominence, spread_factor * spread),
            )
            if peaks.size:
                axis.scatter(
                    time_s[peaks],
                    processed_view[peaks],
                    s=16,
                    color="#c62828",
                    label="Detected peaks",
                    zorder=3,
                )
        axis.set_ylabel(f"{display_name}\n({unit})")
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Signal quality and peak detection | source={variant.data_source} | "
        f"record={variant.record_id} | scenario={variant.scenario}"
    )
    fig.tight_layout()
    return fig


def make_trend_figure(
    timeline: pd.DataFrame, data_source: str, record_id: str, scenario: str
) -> plt.Figure:
    """Plot gated rates and provided SpO2 reference with quality shading."""
    specifications = [
        ("HR", "Heart rate (bpm)"),
        ("PR", "Pulse rate (bpm)"),
        ("RR", "Respiration rate (breaths/min)"),
        ("SpO2_ref", "Provided SpO2 reference (%)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for axis, (column, label) in zip(axes, specifications):
        for row in timeline.itertuples(index=False):
            color = {
                "OK": "#d9f2d9",
                "LOW": "#ffe6b3",
                "MISSING": "#dddddd",
            }[row.overall_quality]
            axis.axvspan(row.start_s, row.end_s, color=color, alpha=0.22, zorder=0)
        axis.plot(
            timeline["start_s"],
            timeline[column],
            marker="o",
            linewidth=1.2,
            label=label,
        )
        axis.set_ylabel(label)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Fused demo trends | source={data_source} | record={record_id} | "
        f"scenario={scenario}"
    )
    fig.tight_layout()
    return fig


def _figure_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()


def _run_fallback() -> None:
    waveforms = load_records("auto")
    waveform = waveforms[0]
    variant, quality, timeline, features, metrics = build_dashboard_data(waveform, "A")
    signal_figure = make_signal_figure(variant, quality)
    trend_figure = make_trend_figure(
        timeline, waveform.data_source, waveform.record_id, "A"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_figure.savefig(OUTPUT_DIR / "dashboard_signal_fallback.png", dpi=150)
    trend_figure.savefig(OUTPUT_DIR / "dashboard_trend_fallback.png", dpi=150)
    timeline.to_csv(OUTPUT_DIR / "dashboard_timeline.csv", index=False)
    features.to_csv(OUTPUT_DIR / "dashboard_feature_table.csv", index=False)
    print(
        f"DASHBOARD FALLBACK | source={waveform.data_source} | "
        f"record={waveform.record_id} | scenario=A"
    )
    print(metrics.to_string(index=False))
    plt.close(signal_figure)
    plt.close(trend_figure)


def _run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="BioSense PH Demo", layout="wide")
    st.title("BioSense PH - signal quality and rate estimation")
    st.warning(
        "Educational demo, not a medical device. Outputs are simulation-only demo flags."
    )
    source_choice = st.sidebar.selectbox(
        "Data source", ("auto", "real", "synthetic"), index=0
    )
    waveforms = load_records(source_choice)
    records = {waveform.record_id: waveform for waveform in waveforms}
    record_id = st.sidebar.selectbox("Record", list(records))
    scenario = st.sidebar.selectbox("Scenario", ("A", "B", "C"))
    waveform = records[record_id]
    variant, signal_quality, timeline, feature_table, metrics = build_dashboard_data(
        waveform, scenario
    )
    st.markdown(
        f"**Data source:** `{waveform.data_source}` &nbsp; "
        f"**Scenario:** `{scenario}` &nbsp; "
        f"**Simulated corruption:** `{variant.simulated_flag}`"
    )
    st.caption(
        "Raw signal -> band-pass processed signal -> calculated rates; "
        "SpO2 is from the provided Numerics reference."
    )
    st.subheader("Signals, processing, and quality shading")
    signal_figure = make_signal_figure(variant, signal_quality)
    st.pyplot(signal_figure, clear_figure=True)
    st.subheader("Fused rates and reference trend")
    trend_figure = make_trend_figure(
        timeline, waveform.data_source, waveform.record_id, scenario
    )
    st.pyplot(trend_figure, clear_figure=True)
    st.caption(
        "Green shading: OK; amber shading: LOW; gray shading: MISSING. "
        "Non-OK values are gated to no-data."
    )
    st.subheader("Rate error and valid-window coverage")
    st.dataframe(metrics, use_container_width=True)
    st.subheader("Simulation-only demo flags")
    st.dataframe(
        timeline[
            [
                "data_source",
                "scenario",
                "start_s",
                "end_s",
                "HR_quality",
                "PR_quality",
                "RR_quality",
                "demo_flag",
                "simulated_flag",
            ]
        ],
        use_container_width=True,
    )
    st.download_button(
        "Download feature table CSV",
        feature_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{record_id}_{scenario}_features.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download fused timeline CSV",
        timeline.to_csv(index=False).encode("utf-8"),
        file_name=f"{record_id}_{scenario}_timeline.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download signal plots PNG",
        _figure_bytes(signal_figure),
        file_name=f"{record_id}_{scenario}_signals.png",
        mime="image/png",
    )
    st.download_button(
        "Download trend plot PNG",
        _figure_bytes(trend_figure),
        file_name=f"{record_id}_{scenario}_trends.png",
        mime="image/png",
    )
    plt.close(signal_figure)
    plt.close(trend_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioSense PH dashboard")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Generate static matplotlib plots and CSVs instead of starting Streamlit",
    )
    arguments = parser.parse_args()
    if arguments.fallback or importlib.util.find_spec("streamlit") is None:
        _run_fallback()
    else:
        _run_streamlit()


if __name__ == "__main__":
    main()
