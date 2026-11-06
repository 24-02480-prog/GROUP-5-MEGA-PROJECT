"""Check expected BIDMC real-data files and basic alignment properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_SIGNAL_COLUMNS = {"Time [s]", "RESP", "PLETH", "II"}
REQUIRED_NUMERICS_COLUMNS = {"Time [s]", "HR", "PULSE", "RESP", "SpO2"}
DURATION_TOLERANCE_S = 2.0


def _report(record_tag: str, check_name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{status} | source=REAL | record={record_tag} | {check_name}{suffix}")
    return passed


def check_record_files(data_dir: Path, record_number: int) -> list[bool]:
    """Run all checks for one Signals/Numerics file pair."""
    record_tag = f"{record_number:02d}"
    signals_path = data_dir / f"bidmc_{record_tag}_Signals.csv"
    numerics_path = data_dir / f"bidmc_{record_tag}_Numerics.csv"
    checks = []
    if not signals_path.is_file():
        checks.append(
            _report(record_tag, "Signals file exists", False, str(signals_path))
        )
    if not numerics_path.is_file():
        checks.append(
            _report(record_tag, "Numerics file exists", False, str(numerics_path))
        )
    if checks:
        return checks

    signals = pd.read_csv(signals_path)
    numerics = pd.read_csv(numerics_path)
    signals.columns = signals.columns.str.strip()
    numerics.columns = numerics.columns.str.strip()

    missing_signals = sorted(REQUIRED_SIGNAL_COLUMNS - set(signals.columns))
    missing_numerics = sorted(REQUIRED_NUMERICS_COLUMNS - set(numerics.columns))
    checks.append(
        _report(
            record_tag,
            "Signals columns",
            not missing_signals,
            f"missing={missing_signals}" if missing_signals else "required columns present",
        )
    )
    checks.append(
        _report(
            record_tag,
            "Numerics columns",
            not missing_numerics,
            f"missing={missing_numerics}" if missing_numerics else "required columns present",
        )
    )
    if missing_signals or missing_numerics:
        return checks

    signal_time = pd.to_numeric(signals["Time [s]"], errors="coerce").to_numpy(float)
    differences = np.diff(signal_time)
    monotonic = (
        len(signal_time) >= 2
        and np.isfinite(signal_time).all()
        and np.all(differences > 0)
    )
    checks.append(_report(record_tag, "Signal time monotonic", monotonic))
    estimated_fs = float(1.0 / np.median(differences)) if monotonic else np.nan
    valid_fs = monotonic and np.isclose(estimated_fs, 125.0, rtol=0, atol=0.5)
    checks.append(
        _report(
            record_tag,
            "Signal sampling rate 125 Hz",
            bool(valid_fs),
            f"observed={estimated_fs:.3f} Hz" if np.isfinite(estimated_fs) else "invalid time",
        )
    )

    numerics_time = pd.to_numeric(
        numerics["Time [s]"], errors="coerce"
    ).to_numpy(dtype=float)
    numerics_monotonic = (
        len(numerics_time) >= 2
        and np.isfinite(numerics_time).all()
        and np.all(np.diff(numerics_time) > 0)
    )
    checks.append(_report(record_tag, "Numerics time monotonic", numerics_monotonic))
    if monotonic and numerics_monotonic:
        signal_start, signal_end = signal_time[0], signal_time[-1]
        numerics_start, numerics_end = numerics_time[0], numerics_time[-1]
        covers_signal = (
            numerics_start <= signal_start + DURATION_TOLERANCE_S
            and numerics_end >= signal_end - DURATION_TOLERANCE_S
            and abs((numerics_end - numerics_start) - (signal_end - signal_start))
            <= DURATION_TOLERANCE_S
        )
        detail = (
            f"signals={signal_end - signal_start:.2f}s; "
            f"numerics={numerics_end - numerics_start:.2f}s"
        )
    else:
        covers_signal, detail = False, "invalid timestamps"
    checks.append(
        _report(record_tag, "Numerics cover signal duration", covers_signal, detail)
    )
    return checks


def check_real_data(data_dir: Path = DATA_DIR) -> bool:
    """Check both expected real records and return whether every check passed."""
    results = [
        result
        for record_number in (1, 2)
        for result in check_record_files(data_dir, record_number)
    ]
    passed = bool(results) and all(results)
    print(f"{'PASS' if passed else 'FAIL'} | source=REAL | overall real-data check")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local BIDMC CSV files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    arguments = parser.parse_args()
    return 0 if check_real_data(arguments.data_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
