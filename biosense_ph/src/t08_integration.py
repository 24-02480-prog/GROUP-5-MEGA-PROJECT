"""Pipeline orchestration for the BioSense PH educational prototype."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .evaluate import evaluate_scenarios
from .scenarios import apply_scenario, severity_scenarios
from .schema import source_labeled_feature_table, validate_feature_table
from .t01_loader import load_records
from .t02_ecg import estimate_ecg_hr
from .t03_ppg import estimate_ppg_pr
from .t04_resp import estimate_resp_rate
from .t05_ml_quality import evaluate_quality_classifiers, scenario_quality_table
from .t05_quality import assess_signal_quality
from .t06_fusion import demonstrate_fusion, fuse_windows
from .t09_ml_rate import evaluate_rate_correction

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _record_number(record_id: str) -> str:
    match = re.search(r"(?:synthetic-)?0?([12])$", str(record_id).lower())
    if not match:
        raise ValueError(f"Unrecognized record identifier: {record_id!r}")
    return match.group(1).zfill(2)


def run_pipeline(
    record_ids: list[str] | None = None,
    scenarios: list[str] | None = None,
    data_source: str = "auto",
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Run loading, scenarios, quality/rate processing, fusion, evaluation, and exports."""
    loaded = load_records(data_source)
    if record_ids is None:
        waveforms = loaded
    else:
        requested = {_record_number(record_id) for record_id in record_ids}
        waveforms = [
            waveform
            for waveform in loaded
            if _record_number(waveform.record_id) in requested
        ]
        if {_record_number(waveform.record_id) for waveform in waveforms} != requested:
            raise ValueError("One or more requested record IDs are unavailable")
    if not waveforms:
        raise ValueError("At least one record must be selected")

    available_scenarios = {"A", "B", "C", *severity_scenarios()}
    selected_scenarios = (
        ["A", "B", "C", *severity_scenarios()]
        if scenarios is None
        else scenarios
    )
    scenario_list = [
        scenario.lower()
        if scenario.lower() in severity_scenarios()
        else scenario.upper()
        for scenario in selected_scenarios
    ]
    if not scenario_list or any(
        scenario not in available_scenarios for scenario in scenario_list
    ):
        raise ValueError("scenarios must contain A, B, C, or configured severity IDs")
    if len(set(scenario_list)) != len(scenario_list):
        raise ValueError("scenarios must not contain duplicates")

    output_dir.mkdir(parents=True, exist_ok=True)
    fused_tables = []
    baseline_feature_tables = []
    pass_fail_tables = []
    demo_tables = []
    for waveform in waveforms:
        print(
            f"PIPELINE | source={waveform.data_source} | record={waveform.record_id}"
        )
        pass_fail_tables.append(scenario_quality_table(waveform, output_dir))
        demo_tables.append(demonstrate_fusion(waveform, output_dir))
        for scenario_label in scenario_list:
            variant = apply_scenario(waveform, scenario_label)
            rate_tables = [
                estimate_ecg_hr(variant),
                estimate_ppg_pr(variant),
                estimate_resp_rate(variant),
            ]
            signal_quality = {
                name: assess_signal_quality(variant, name)
                for name in ("II", "PLETH", "RESP")
            }
            timeline, feature_table = fuse_windows(
                variant,
                rate_tables,
                signal_quality,
                scenario=scenario_label,
                simulated_flag=variant.simulated_flag,
            )
            fused_tables.append(timeline)
            if scenario_label == "A":
                validate_feature_table(feature_table)
                baseline_feature_tables.append(feature_table)

    if not baseline_feature_tables:
        for waveform in waveforms:
            baseline = apply_scenario(waveform, "A")
            baseline_rates = [
                estimate_ecg_hr(baseline),
                estimate_ppg_pr(baseline),
                estimate_resp_rate(baseline),
            ]
            baseline_quality = {
                name: assess_signal_quality(baseline, name)
                for name in ("II", "PLETH", "RESP")
            }
            _, baseline_table = fuse_windows(
                baseline,
                baseline_rates,
                baseline_quality,
                scenario="A",
            )
            baseline_feature_tables.append(baseline_table)

    fused_timeline = pd.concat(fused_tables, ignore_index=True)
    feature_table = pd.concat(baseline_feature_tables, ignore_index=True)
    feature_table = source_labeled_feature_table(
        feature_table,
        {waveform.record_id: waveform.data_source for waveform in waveforms},
    )
    pass_fail = pd.concat(pass_fail_tables, ignore_index=True)
    demos = pd.concat(demo_tables, ignore_index=True)
    fused_timeline.to_csv(output_dir / "fused_timeline.csv", index=False)
    feature_table.to_csv(output_dir / "final_feature_table.csv", index=False)
    pass_fail.to_csv(output_dir / "scenario_quality_pass_fail.csv", index=False)
    demos.to_csv(output_dir / "fusion_scenario_demos.csv", index=False)

    classifier_metrics, confusion, importance = evaluate_quality_classifiers(
        loaded, output_dir
    )
    errors = evaluate_scenarios(waveforms, output_dir, scenario_list)
    (
        rate_correction_metrics,
        rate_correction_predictions,
        corrected_rate_features,
        peak_rate_features,
    ) = evaluate_rate_correction(loaded, output_dir)
    print(
        "PIPELINE COMPLETE | source="
        + ",".join(sorted({waveform.data_source for waveform in waveforms}))
        + f" | records={len(waveforms)} | scenarios={','.join(scenario_list)}"
    )
    return {
        "fused_timeline": fused_timeline,
        "feature_table": feature_table,
        "scenario_quality": pass_fail,
        "fusion_demos": demos,
        "classifier_metrics": classifier_metrics,
        "confusion_matrix": confusion,
        "feature_importance": importance,
        "error_table": errors,
        "ml_rate_correction_metrics": rate_correction_metrics,
        "ml_rate_correction_predictions": rate_correction_predictions,
        "ml_corrected_feature_table": corrected_rate_features,
        "peak_based_feature_table": peak_rate_features,
    }
