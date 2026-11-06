from __future__ import annotations

from check_real_data import check_real_data, check_record_files
from src.synthetic_data import generate_synthetic_record


def _write_record(data_dir, number):
    record = generate_synthetic_record(number)
    record.signals.to_csv(
        data_dir / f"bidmc_{number:02d}_Signals.csv", index=False
    )
    record.numerics.to_csv(
        data_dir / f"bidmc_{number:02d}_Numerics.csv", index=False
    )


def test_checker_passes_valid_pair_and_both_records(tmp_path):
    _write_record(tmp_path, 1)
    _write_record(tmp_path, 2)
    assert all(check_record_files(tmp_path, 1))
    assert check_real_data(tmp_path)


def test_checker_fails_missing_files_and_wrong_sampling_rate(tmp_path):
    assert not check_real_data(tmp_path)
    _write_record(tmp_path, 1)
    path = tmp_path / "bidmc_01_Signals.csv"
    import pandas as pd

    frame = pd.read_csv(path)
    frame = frame.iloc[::2].reset_index(drop=True)
    frame.to_csv(path, index=False)
    assert not all(check_record_files(tmp_path, 1))
