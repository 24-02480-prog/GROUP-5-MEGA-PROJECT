"""Rate fusion, quality gating, and simulation-only demonstration flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_config
from .scenarios import severity_scenarios
from .t01_loader import Waveform
from .schema import FEATURE_COLUMNS, source_record_id, validate_feature_table

DEFAULT_THRESHOLDS = {
    "hr_min_bpm": 50.0,
    "hr_max_bpm": 110.0,
    "pr_min_bpm": 50.0,
    "pr_max_bpm": 110.0,
    "rr_min_breaths_min": 8.0,
    "rr_max_breaths_min": 30.0,
    "spo2_ref_min_percent": 92.0,
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
QUALITY_PRIORITY = {"OK": 0, "LOW": 1, "MISSING": 2}


def _combine_quality(*labels: str) -> str:
    return max(labels, key=QUALITY_PRIORITY.__getitem__)


def _reference_value(waveform: Waveform, start_s: float, end_s: float) -> float:
    references = waveform.numerics
    times = references["Time [s]"].to_numpy(dtype=float)
    window = (times >= start_s) & (times < end_s)
    values = pd.to_numeric(
        references.loc[window, "SpO2"], errors="coerce"
    ).to_numpy(dtype=float)
    return float(np.nanmean(values)) if values.size and np.isfinite(values).any() else np.nan


def fuse_windows(
    waveform: Waveform,
    rate_tables: list[pd.DataFrame],
    signal_quality_tables: dict[str, pd.DataFrame] | None = None,
    thresholds: dict[str, float] | None = None,
    scenario: str = "A",
    demo_case: str = "BASELINE",
    simulated_flag: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate estimated rates, merge provided SpO2 references, and assign demo flags."""
    config = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(config)
        if unknown:
            raise ValueError(f"Unknown demo threshold(s): {sorted(unknown)}")
        config.update(thresholds)
    if scenario not in {"A", "B", "C", *severity_scenarios()}:
        raise ValueError("scenario must be A, B, C, or a configured severity")
    tables_by_feature: dict[str, pd.DataFrame] = {}
    for table in rate_tables:
        validate_feature_table(table)
        for feature in table["feature"].unique():
            if feature not in {"HR", "PR", "RR"}:
                raise ValueError(f"Unsupported rate feature in fusion: {feature}")
            tables_by_feature[feature] = table[table["feature"] == feature].copy()
    if set(tables_by_feature) != {"HR", "PR", "RR"}:
        raise ValueError("Fusion requires HR, PR, and RR feature tables")

    signal_map = {"HR": "II", "PR": "PLETH", "RR": "RESP"}
    indexes: dict[str, dict[float, object]] = {}
    for feature, table in tables_by_feature.items():
        expected_record_id = source_record_id(
            waveform.record_id, waveform.data_source
        )
        if not table["record_id"].isin(
            {waveform.record_id, expected_record_id}
        ).all():
            raise ValueError(f"{feature} table record_id does not match waveform")
        indexes[feature] = {
            float(row.start_s): row for row in table.itertuples(index=False)
        }
    quality_indexes: dict[str, dict[float, str]] = {}
    for signal_name, table in (signal_quality_tables or {}).items():
        quality_indexes[signal_name] = {
            float(row.start_s): str(row.quality)
            for row in table.itertuples(index=False)
        }

    output_rows = []
    feature_rows = []
    rate_units = {
        "HR": "bpm",
        "PR": "bpm",
        "RR": "breaths/min",
        "HR_PR_diff": "bpm",
    }
    rate_sources = {
        "HR": "calculated_feature:processed_ECG_from_raw_II",
        "PR": "calculated_feature:processed_PPG_from_raw_PLETH",
        "RR": "calculated_feature:processed_RESP_from_raw_RESP",
    }
    for feature in ("HR", "PR", "RR"):
        validate_feature_table(tables_by_feature[feature])
    window_starts = sorted(indexes["HR"])
    for start_s in window_starts:
        hr_row = indexes["HR"].get(start_s)
        if hr_row is None:
            continue
        end_s = float(hr_row.end_s)
        values: dict[str, float] = {}
        qualities: dict[str, str] = {}
        for feature in ("HR", "PR", "RR"):
            row = indexes[feature].get(start_s)
            if row is None:
                values[feature], qualities[feature] = np.nan, "MISSING"
                continue
            estimator_quality = str(row.quality)
            signal_quality = quality_indexes.get(signal_map[feature], {}).get(
                start_s, "OK"
            )
            quality = _combine_quality(estimator_quality, signal_quality)
            value = float(row.value) if pd.notna(row.value) else np.nan
            values[feature] = value if quality == "OK" else np.nan
            qualities[feature] = quality

        if np.isfinite(values["HR"]) and np.isfinite(values["PR"]):
            hr_pr_difference = abs(values["HR"] - values["PR"])
            cross_check_quality = (
                "LOW"
                if hr_pr_difference
                > float(
                    get_config()["cross_check"]["hr_pr_max_difference_bpm"]
                )
                else "OK"
            )
        else:
            hr_pr_difference, cross_check_quality = np.nan, "MISSING"
        values["HR_PR_diff"] = hr_pr_difference
        qualities["HR_PR_diff"] = cross_check_quality

        spo2 = _reference_value(waveform, start_s, end_s)
        spo2_quality = "OK" if np.isfinite(spo2) else "MISSING"
        values["SpO2_ref"] = spo2 if spo2_quality == "OK" else np.nan
        qualities["SpO2_ref"] = spo2_quality
        overall_quality = _combine_quality(*qualities.values())

        if overall_quality != "OK":
            demo_flag = "NO-DATA"
        elif (
            values["HR"] < config["hr_min_bpm"]
            or values["HR"] > config["hr_max_bpm"]
            or values["PR"] < config["pr_min_bpm"]
            or values["PR"] > config["pr_max_bpm"]
            or values["RR"] < config["rr_min_breaths_min"]
            or values["RR"] > config["rr_max_breaths_min"]
            or values["SpO2_ref"] < config["spo2_ref_min_percent"]
        ):
            demo_flag = "CAUTION-DEMO"
        else:
            demo_flag = "NORMAL-DEMO"

        simulation_applied = (
            scenario != "A" or demo_case != "BASELINE"
            if simulated_flag is None
            else simulated_flag
        )
        output_rows.append(
            {
                "record_id": source_record_id(
                    waveform.record_id, waveform.data_source
                ),
                "data_source": waveform.data_source,
                "scenario": scenario,
                "demo_case": demo_case,
                "simulated_flag": simulation_applied,
                "start_s": start_s,
                "end_s": end_s,
                **values,
                "HR_quality": qualities["HR"],
                "PR_quality": qualities["PR"],
                "RR_quality": qualities["RR"],
                "SpO2_quality": qualities["SpO2_ref"],
                "overall_quality": overall_quality,
                "HR_PR_diff": values["HR_PR_diff"],
                "HR_PR_diff_quality": qualities["HR_PR_diff"],
                "HR_PR_diff_lineage": (
                    "calculated_feature:absolute_difference_of_HR_and_PR"
                ),
                "demo_flag": demo_flag,
                "HR_lineage": rate_sources["HR"],
                "PR_lineage": rate_sources["PR"],
                "RR_lineage": rate_sources["RR"],
                "SpO2_ref_lineage": "provided_reference:Numerics.SpO2",
                "flag_lineage": "simulation-only demo flags",
            }
        )
        for feature in ("HR", "PR", "RR", "HR_PR_diff", "SpO2_ref"):
            feature_rows.append(
                {
                    "record_id": source_record_id(
                        waveform.record_id, waveform.data_source
                    ),
                    "start_s": start_s,
                    "end_s": end_s,
                    "feature": feature,
                    "value": values[feature],
                    "unit": rate_units.get(feature, "%"),
                    "quality": qualities[feature],
                }
            )
        feature_rows.extend(
            [
                {
                    "record_id": source_record_id(
                        waveform.record_id, waveform.data_source
                    ),
                    "start_s": start_s,
                    "end_s": end_s,
                    "feature": "quality",
                    "value": overall_quality,
                    "unit": "label",
                    "quality": overall_quality,
                },
                {
                    "record_id": source_record_id(
                        waveform.record_id, waveform.data_source
                    ),
                    "start_s": start_s,
                    "end_s": end_s,
                    "feature": "scenario_flag",
                    "value": demo_flag,
                    "unit": "label",
                    "quality": overall_quality,
                },
            ]
        )

    timeline = pd.DataFrame(output_rows)
    feature_table = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    validate_feature_table(feature_table)
    print(
        f"T06 FUSION | source={waveform.data_source} | record={waveform.record_id} "
        f"| scenario={scenario} | case={demo_case}"
    )
    return timeline, feature_table


def demonstrate_fusion(
    waveform: Waveform, output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """Export A/B/C examples plus a clearly labeled threshold-only caution demo."""
    from .scenarios import apply_scenario
    from .t02_ecg import estimate_ecg_hr
    from .t03_ppg import estimate_ppg_pr
    from .t04_resp import estimate_resp_rate
    from .t05_quality import assess_signal_quality

    demo_rows = []
    for scenario_label in ("A", "B", "C"):
        variant = apply_scenario(waveform, scenario_label)
        rates = [
            estimate_ecg_hr(variant),
            estimate_ppg_pr(variant),
            estimate_resp_rate(variant),
        ]
        quality = {
            signal_name: assess_signal_quality(variant, signal_name)
            for signal_name in ("II", "PLETH", "RESP")
        }
        timeline, _ = fuse_windows(
            variant,
            rates,
            quality,
            scenario=scenario_label,
            simulated_flag=variant.simulated_flag,
        )
        demo_rows.append(timeline)
        if scenario_label == "A":
            observed_hr = pd.to_numeric(timeline["HR"], errors="coerce").dropna()
            if observed_hr.empty:
                raise ValueError("Cannot create caution demo without an available HR")
            caution_threshold = float(observed_hr.median() - 0.1)
            caution_timeline, _ = fuse_windows(
                variant,
                rates,
                quality,
                thresholds={"hr_max_bpm": caution_threshold},
                scenario="A",
                demo_case="THRESHOLD-OVERRIDE-CAUTION",
                simulated_flag=True,
            )
            demo_rows.append(caution_timeline)

    result = pd.concat(demo_rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "fusion_scenario_demos.csv", index=False)
    print("T06 DEMO FLAGS | source=" + waveform.data_source)
    print(
        result.groupby(["scenario", "demo_case", "demo_flag"], dropna=False)
        .size()
        .rename("windows")
        .reset_index()
        .to_string(index=False)
    )
    return result
