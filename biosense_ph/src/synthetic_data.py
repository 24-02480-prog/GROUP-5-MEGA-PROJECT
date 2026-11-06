"""Deterministic BIDMC-like stand-in data for development and testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticRecord:
    """Synthetic signals and their 1 Hz reference numerics."""

    record_id: str
    signals: pd.DataFrame
    numerics: pd.DataFrame


def _instantaneous_rate(
    time_s: np.ndarray, base_bpm: float, amplitude_bpm: float, period_s: float, phase: float
) -> np.ndarray:
    return base_bpm + amplitude_bpm * np.sin(2 * np.pi * time_s / period_s + phase)


def _beat_times(time_s: np.ndarray, rate_bpm: np.ndarray) -> np.ndarray:
    phase_cycles = np.concatenate(
        ([0.0], np.cumsum((rate_bpm[1:] + rate_bpm[:-1]) * 0.5 * np.diff(time_s) / 60))
    )
    beat_indices = np.flatnonzero(np.diff(np.floor(phase_cycles)) > 0) + 1
    return time_s[beat_indices]


def _pulse_wave(
    time_s: np.ndarray,
    beat_times: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    signal = np.zeros_like(time_s)
    for beat in beat_times:
        left = max(0, np.searchsorted(time_s, beat - 0.45))
        right = min(len(time_s), np.searchsorted(time_s, beat + 0.8))
        relative = time_s[left:right] - beat
        signal[left:right] += amplitude * np.exp(-0.5 * (relative / 0.035) ** 2)
        delayed = relative - 0.24
        signal[left:right] += 0.20 * amplitude * np.exp(
            -0.5 * (delayed / 0.075) ** 2
        )
    signal += 0.025 * np.sin(2 * np.pi * time_s / 7.5)
    signal += rng.normal(0, 0.012, size=time_s.size)
    return signal


def generate_synthetic_record(
    record_number: int,
    duration_s: int = 480,
    fs: int = 125,
) -> SyntheticRecord:
    """Generate one deterministic eight-minute BIDMC-like record."""
    if record_number not in (1, 2):
        raise ValueError("record_number must be 1 or 2")
    if duration_s < 60 or fs != 125:
        raise ValueError("duration_s must be at least 60 and fs must be 125 Hz")

    rng = np.random.default_rng(4050 + record_number)
    time_s = np.arange(0, duration_s, 1 / fs, dtype=np.float64)
    phase = 0.3 if record_number == 1 else 1.1
    heart_rate = _instantaneous_rate(
        time_s, 75.0 if record_number == 1 else 82.0, 3.0, 150.0, phase
    )
    beats = _beat_times(time_s, heart_rate)

    ecg = np.zeros_like(time_s)
    for beat in beats:
        left = max(0, np.searchsorted(time_s, beat - 0.18))
        right = min(len(time_s), np.searchsorted(time_s, beat + 0.18))
        relative = time_s[left:right] - beat
        ecg[left:right] += 1.0 * np.exp(-0.5 * (relative / 0.012) ** 2)
        ecg[left:right] -= 0.12 * np.exp(-0.5 * ((relative + 0.035) / 0.018) ** 2)
        ecg[left:right] -= 0.18 * np.exp(-0.5 * ((relative - 0.055) / 0.025) ** 2)
    ecg += 0.08 * np.sin(2 * np.pi * time_s / 8.0)
    ecg += rng.normal(0, 0.018, size=time_s.size)

    pulse = _pulse_wave(time_s, beats, rng, amplitude=0.8 if record_number == 1 else 0.7)
    respiration_rate = _instantaneous_rate(
        time_s, 15.0 if record_number == 1 else 17.0, 0.8, 180.0, phase
    )
    respiratory_phase = np.concatenate(
        ([0.0], np.cumsum(
            (respiration_rate[1:] + respiration_rate[:-1])
            * 0.5
            * np.diff(time_s)
            / 60
        ))
    )
    resp = 0.45 * np.sin(2 * np.pi * respiratory_phase)
    resp += 0.015 * np.sin(2 * np.pi * time_s / 13.0)
    resp += rng.normal(0, 0.008, size=time_s.size)

    signals = pd.DataFrame(
        {
            " Time [s]": time_s,
            " RESP": resp,
            " PLETH": pulse,
            " V": 0.0 * time_s + rng.normal(0, 0.005, size=time_s.size),
            " AVR": 0.0 * time_s + rng.normal(0, 0.005, size=time_s.size),
            " II": ecg,
        }
    )

    second_times = np.arange(0, duration_s, dtype=float)
    heart_ref = _instantaneous_rate(
        second_times, 75.0 if record_number == 1 else 82.0, 3.0, 150.0, phase
    )
    respiration_ref = _instantaneous_rate(
        second_times,
        15.0 if record_number == 1 else 17.0,
        0.8,
        180.0,
        phase,
    )
    numerics = pd.DataFrame(
        {
            " Time [s]": second_times,
            " HR": heart_ref,
            " PULSE": heart_ref.copy(),
            " RESP": respiration_ref,
            " SpO2": np.full(second_times.size, 97.0 if record_number == 1 else 96.0),
        }
    )
    return SyntheticRecord(
        record_id=f"synthetic-{record_number:02d}",
        signals=signals,
        numerics=numerics,
    )


def generate_synthetic_records() -> list[SyntheticRecord]:
    """Return both fixed-seed stand-in records."""
    return [generate_synthetic_record(1), generate_synthetic_record(2)]
