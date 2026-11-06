"""Load and align real BIDMC CSVs or deterministic synthetic stand-ins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .synthetic_data import SyntheticRecord, generate_synthetic_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REQUIRED_SIGNALS = {"Time [s]", "RESP", "PLETH", "II"}
REQUIRED_NUMERICS = {"Time [s]", "HR", "PULSE", "RESP", "SpO2"}


@dataclass(frozen=True)
class Waveform:
    """Aligned waveform and reference data with explicit source provenance."""

    record_id: str
    time_s: np.ndarray
    fs: float
    signals: dict[str, np.ndarray]
    numerics: pd.DataFrame
    data_source: str


def _record_number(record_id: str) -> int:
    normalized = str(record_id).lower().replace("synthetic-", "").zfill(2)
    if normalized not in {"01", "02"}:
        raise ValueError("record_id must identify record 01 or 02")
    return int(normalized)


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def _estimate_fs(time_s: np.ndarray) -> float:
    if time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("Signal time column must contain at least two samples")
    differences = np.diff(time_s)
    if not np.isfinite(time_s).all() or np.any(differences <= 0):
        raise ValueError("Signal timestamps must be finite and strictly increasing")
    return float(1.0 / np.median(differences))


def _waveform_from_frames(
    record_id: str,
    signals_frame: pd.DataFrame,
    numerics: pd.DataFrame,
    data_source: str,
) -> Waveform:
    missing_signals = sorted(REQUIRED_SIGNALS - set(signals_frame.columns))
    missing_numerics = sorted(REQUIRED_NUMERICS - set(numerics.columns))
    if missing_signals:
        raise ValueError(f"Signal CSV is missing required columns: {missing_signals}")
    if missing_numerics:
        raise ValueError(f"Numerics CSV is missing required columns: {missing_numerics}")
    time_s = pd.to_numeric(signals_frame["Time [s]"], errors="coerce").to_numpy(float)
    fs = _estimate_fs(time_s)
    if not np.isclose(fs, 125.0, rtol=0, atol=0.5):
        raise ValueError(f"Record {record_id} sampling rate is {fs:.3f} Hz, expected 125 Hz")
    if not np.isfinite(numerics["Time [s]"].to_numpy(dtype=float)).all():
        raise ValueError("Numerics timestamps must be finite")
    if np.any(np.diff(numerics["Time [s]"].to_numpy(dtype=float)) <= 0):
        raise ValueError("Numerics timestamps must be strictly increasing")
    signal_names = ("RESP", "PLETH", "V", "AVR", "II")
    output_signals: dict[str, np.ndarray] = {}
    for name in signal_names:
        if name in signals_frame:
            output_signals[name] = pd.to_numeric(
                signals_frame[name], errors="coerce"
            ).to_numpy(dtype=float)
    return Waveform(
        record_id=record_id,
        time_s=time_s,
        fs=fs,
        signals=output_signals,
        numerics=numerics.copy(),
        data_source=data_source,
    )


def _load_real(record_number: int) -> Waveform:
    record_tag = f"{record_number:02d}"
    signals_path = DATA_DIR / f"bidmc_{record_tag}_Signals.csv"
    numerics_path = DATA_DIR / f"bidmc_{record_tag}_Numerics.csv"
    if not signals_path.is_file() or not numerics_path.is_file():
        missing = [str(path) for path in (signals_path, numerics_path) if not path.is_file()]
        raise FileNotFoundError("Required real-data file(s) not found: " + ", ".join(missing))
    return _waveform_from_frames(
        record_tag, _read_csv(signals_path), _read_csv(numerics_path), "REAL"
    )


def _load_synthetic(record_number: int) -> Waveform:
    record: SyntheticRecord = generate_synthetic_record(record_number)
    signals = record.signals.copy()
    numerics = record.numerics.copy()
    signals.columns = signals.columns.str.strip()
    numerics.columns = numerics.columns.str.strip()
    return _waveform_from_frames(
        record.record_id, signals, numerics, "SYNTHETIC"
    )


def _summarize(waveforms: list[Waveform]) -> None:
    rows = []
    for waveform in waveforms:
        signal_nan_counts = {
            name: int(np.isnan(values).sum())
            for name, values in waveform.signals.items()
        }
        duration = float(waveform.time_s[-1] - waveform.time_s[0])
        rows.append(
            {
                "record": waveform.record_id,
                "source": waveform.data_source,
                "duration_s": round(duration, 2),
                "fs_Hz": round(waveform.fs, 3),
                "signals": ",".join(waveform.signals),
                "signal_NaNs": signal_nan_counts,
                "reference_columns": ",".join(waveform.numerics.columns),
            }
        )
    print("DATA SUMMARY")
    print(pd.DataFrame(rows).to_string(index=False))


def load_records(data_source: str = "auto") -> list[Waveform]:
    """Load both records from real files or synthetic fallback data.

    ``auto`` uses real data only when all four expected CSV files are present.
    """
    source = data_source.lower()
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("data_source must be one of: auto, real, synthetic")
    real_paths = [
        DATA_DIR / f"bidmc_{record:02d}_{kind}.csv"
        for record in (1, 2)
        for kind in ("Signals", "Numerics")
    ]
    real_available = all(path.is_file() for path in real_paths)
    if source == "real" and not real_available:
        missing = [str(path) for path in real_paths if not path.is_file()]
        raise FileNotFoundError(
            "REAL DATA NOT FOUND - missing required file(s): " + ", ".join(missing)
        )
    if source == "auto" and not real_available:
        warning = "REAL DATA NOT FOUND - USING SYNTHETIC DATA"
        print(warning)
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        selected_source = "synthetic"
    else:
        selected_source = "real" if source in {"auto", "real"} else "synthetic"

    waveforms = [
        _load_real(number) if selected_source == "real" else _load_synthetic(number)
        for number in (1, 2)
    ]
    _summarize(waveforms)
    return waveforms


def load_record(record_id: str, data_source: str = "auto") -> Waveform:
    """Load one record, accepting ``01``/``02`` and synthetic aliases."""
    record_number = _record_number(record_id)
    source = data_source.lower()
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("data_source must be one of: auto, real, synthetic")
    real_paths = [
        DATA_DIR / f"bidmc_{record_number:02d}_{kind}.csv"
        for kind in ("Signals", "Numerics")
    ]
    if source == "real":
        waveform = _load_real(record_number)
    elif source == "synthetic":
        waveform = _load_synthetic(record_number)
    elif all(path.is_file() for path in real_paths):
        waveform = _load_real(record_number)
    else:
        print("REAL DATA NOT FOUND - USING SYNTHETIC DATA")
        warnings.warn(
            "REAL DATA NOT FOUND - USING SYNTHETIC DATA",
            RuntimeWarning,
            stacklevel=2,
        )
        waveform = _load_synthetic(record_number)
    _summarize([waveform])
    return waveform


if __name__ == "__main__":
    load_records()
