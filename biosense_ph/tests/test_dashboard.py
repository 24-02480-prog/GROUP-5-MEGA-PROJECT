from __future__ import annotations

import matplotlib.pyplot as plt

from src.t07_dashboard import (
    build_dashboard_data,
    make_signal_figure,
    make_trend_figure,
)


def test_dashboard_builds_source_labeled_tables_and_figures(waveform):
    variant, quality, timeline, features, metrics = build_dashboard_data(
        waveform, "A"
    )
    assert variant.data_source == "SYNTHETIC"
    assert timeline["data_source"].eq("SYNTHETIC").all()
    assert set(metrics["rate"]) == {"HR", "PR", "RR"}
    assert not features.empty
    signal_figure = make_signal_figure(variant, quality)
    trend_figure = make_trend_figure(
        timeline, variant.data_source, variant.record_id, "A"
    )
    assert len(signal_figure.axes) == 3
    assert len(trend_figure.axes) == 4
    plt.close(signal_figure)
    plt.close(trend_figure)


def test_dashboard_gates_missing_scenario_to_no_data(waveform):
    variant, _, timeline, _, _ = build_dashboard_data(waveform, "C")
    assert variant.simulated_flag is True
    assert timeline["demo_flag"].eq("NO-DATA").all()
