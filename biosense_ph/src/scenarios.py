"""Deterministic clean and corrupted signal scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import get_config
from .t01_loader import Waveform

SCENARIO_LABELS = {"A": "unmodified", "B": "noise-spikes", "C": "flatline-missing"}


def severity_scenarios() -> dict[str, dict[str, float | int | str]]:
    """Return configured, deterministic corruption levels keyed by scenario ID."""
    scenario_config = get_config()["scenario"]
    levels: dict[str, dict[str, float | int | str]] = {}
    for snr_db in scenario_config["noise_snr_db"]:
        levels[f"noise_snr_{snr_db}db"] = {"kind": "noise_snr", "snr_db": snr_db}
    for count in scenario_config["spike_counts_per_window"]:
        levels[f"spike_burst_{count}"] = {"kind": "spike_burst", "count": count}
    for percentage in scenario_config["dropout_percentages"]:
        levels[f"dropout_{percentage}pct"] = {
            "kind": "dropout",
            "percentage": percentage,
        }
    for name, fraction in scenario_config["baseline_drift"]["levels"].items():
        levels[f"baseline_drift_{name}"] = {
            "kind": "baseline_drift",
            "fraction": fraction,
            "frequency_hz": scenario_config["baseline_drift"]["frequency_hz"],
        }
    for fraction in scenario_config["attenuation_fractions"]:
        percent = int(round(float(fraction) * 100))
        levels[f"attenuation_{percent}pct"] = {
            "kind": "attenuation",
            "fraction": fraction,
        }
    return levels


def _apply_severity(
    waveform: Waveform, scenario: str, definition: dict[str, float | int | str]
) -> dict[str, np.ndarray]:
    config = get_config()["scenario"]
    window_seconds = float(config["window_seconds"])
    samples_per_window = int(round(window_seconds * waveform.fs))
    seed = int(config["random_seed"])
    signals: dict[str, np.ndarray] = {}
    for signal_name, original in waveform.signals.items():
        values = np.asarray(original, dtype=float).copy()
        for window_index, start in enumerate(range(0, values.size, samples_per_window)):
            end = min(start + samples_per_window, values.size)
            segment = values[start:end]
            finite = np.isfinite(segment)
            if not finite.any():
                continue
            clean = segment[finite].copy()
            scale = max(float(np.std(clean)), 1e-8)
            rng = np.random.default_rng(
                seed
                + sum(ord(char) for char in waveform.record_id + signal_name + scenario)
                + window_index
            )
            kind = definition["kind"]
            if kind == "noise_snr":
                snr_db = float(definition["snr_db"])
                noise_std = scale / (10 ** (snr_db / 20))
                segment[finite] += rng.normal(0.0, noise_std, size=clean.size)
            elif kind == "spike_burst":
                count = min(int(definition["count"]), clean.size)
                indexes = rng.choice(clean.size, size=count, replace=False)
                local_indexes = np.flatnonzero(finite)[indexes]
                segment[local_indexes] += (
                    rng.choice([-1.0, 1.0], size=count)
                    * float(config["spike_amplitude_std"])
                    * scale
                )
            elif kind == "dropout":
                dropout_count = int(round(
                    clean.size * float(definition["percentage"]) / 100
                ))
                if dropout_count:
                    dropout_start = int(rng.integers(0, max(1, clean.size - dropout_count + 1)))
                    local_indexes = np.flatnonzero(finite)
                    segment[
                        local_indexes[dropout_start : dropout_start + dropout_count]
                    ] = np.nan
            elif kind == "baseline_drift":
                positions = np.arange(segment.size, dtype=float)
                drift = (
                    scale
                    * float(definition["fraction"])
                    * np.sin(
                        2
                        * np.pi
                        * float(definition["frequency_hz"])
                        * (positions + start)
                        / waveform.fs
                    )
                )
                segment[finite] += drift[finite]
            elif kind == "attenuation":
                midpoint = float(np.mean(clean))
                segment[finite] = midpoint + (
                    segment[finite] - midpoint
                ) * float(definition["fraction"])
            else:
                raise ValueError(f"Unknown severity corruption kind: {kind!r}")
        signals[signal_name] = values
    return signals


@dataclass(frozen=True)
class ScenarioRecord:
    """A waveform variant with its simulation scenario tracked explicitly."""

    record_id: str
    data_source: str
    time_s: np.ndarray
    fs: float
    signals: dict[str, np.ndarray]
    numerics: object
    scenario: str
    simulated_flag: bool


def apply_scenario(waveform: Waveform, scenario: str) -> ScenarioRecord:
    """Return a fixed-seed original or severity variant without mutating input."""
    scenario = scenario.lower() if scenario.lower() in severity_scenarios() else scenario.upper()
    severity_config = severity_scenarios()
    if scenario in severity_config:
        return ScenarioRecord(
            record_id=waveform.record_id,
            data_source=waveform.data_source,
            time_s=waveform.time_s.copy(),
            fs=waveform.fs,
            signals=_apply_severity(waveform, scenario, severity_config[scenario]),
            numerics=waveform.numerics.copy(),
            scenario=scenario,
            simulated_flag=True,
        )
    if scenario not in SCENARIO_LABELS:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected A, B, C, or a configured severity"
        )

    variant: dict[str, np.ndarray] = {}
    for signal_name, original in waveform.signals.items():
        values = np.asarray(original, dtype=float).copy()
        if scenario == "B":
            rng = np.random.default_rng(
                40500 + sum(ord(char) for char in waveform.record_id + signal_name)
            )
            scale = max(float(np.nanstd(values)), 0.02)
            values += rng.normal(0, 0.8 * scale, size=values.size)
            spike_count = max(5, int(len(values) / waveform.fs / 3))
            spike_indexes = rng.choice(values.size, size=spike_count, replace=False)
            values[spike_indexes] += rng.choice([-1.0, 1.0], size=spike_count) * 100 * scale
        elif scenario == "C":
            samples_per_window = int(round(30 * waveform.fs))
            for window_start in range(0, values.size, samples_per_window):
                flat_start = window_start + int(5 * waveform.fs)
                flat_end = min(flat_start + int(10 * waveform.fs), values.size)
                if flat_start < flat_end:
                    baseline = values[max(0, flat_start - 1)]
                    values[flat_start:flat_end] = baseline
                missing_start = window_start + int(18 * waveform.fs)
                missing_end = min(missing_start + int(8 * waveform.fs), values.size)
                if missing_start < missing_end:
                    values[missing_start:missing_end] = np.nan
        variant[signal_name] = values

    return ScenarioRecord(
        record_id=waveform.record_id,
        data_source=waveform.data_source,
        time_s=waveform.time_s.copy(),
        fs=waveform.fs,
        signals=variant,
        numerics=waveform.numerics.copy(),
        scenario=scenario,
        simulated_flag=scenario != "A",
    )


def apply_quality_example(waveform: Waveform, example: str) -> ScenarioRecord:
    """Create one isolated corruption example for the rule-based pass/fail table."""
    example = example.lower()
    if example == "flatline-missing":
        return apply_scenario(waveform, "C")
    if example not in {"noise-only", "spikes-only"}:
        raise ValueError(
            "example must be noise-only, spikes-only, or flatline-missing"
        )

    variant = apply_scenario(waveform, "A")
    signals = {}
    for signal_name, original in variant.signals.items():
        values = original.copy()
        rng = np.random.default_rng(
            40700 + sum(ord(char) for char in waveform.record_id + signal_name + example)
        )
        scale = max(float(np.nanstd(values)), 0.02)
        if example == "noise-only":
            values += rng.normal(0, 8.0 * scale, size=values.size)
        else:
            count = max(5, int(len(values) / waveform.fs / 3))
            indexes = rng.choice(values.size, size=count, replace=False)
            values[indexes] += (
                rng.choice([-1.0, 1.0], size=count) * 100.0 * scale
            )
        signals[signal_name] = values

    return ScenarioRecord(
        record_id=variant.record_id,
        data_source=variant.data_source,
        time_s=variant.time_s.copy(),
        fs=variant.fs,
        signals=signals,
        numerics=variant.numerics.copy(),
        scenario="B",
        simulated_flag=True,
    )
