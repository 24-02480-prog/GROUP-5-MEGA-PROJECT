from __future__ import annotations

import numpy as np

from src.synthetic_data import generate_synthetic_record, generate_synthetic_records


def test_generator_matches_leading_space_csv_headers_and_duration():
    record = generate_synthetic_record(1)
    assert record.signals.columns.tolist() == [
        " Time [s]",
        " RESP",
        " PLETH",
        " V",
        " AVR",
        " II",
    ]
    assert record.numerics.columns.tolist() == [
        " Time [s]",
        " HR",
        " PULSE",
        " RESP",
        " SpO2",
    ]
    assert len(record.signals) == 480 * 125
    assert len(record.numerics) == 480


def test_generator_uses_fixed_seeds_and_distinct_records():
    first_run = generate_synthetic_records()
    second_run = generate_synthetic_records()
    assert [record.record_id for record in first_run] == [
        "synthetic-01",
        "synthetic-02",
    ]
    np.testing.assert_array_equal(
        first_run[0].signals[" II"], second_run[0].signals[" II"]
    )
    assert not np.array_equal(
        first_run[0].signals[" II"], first_run[1].signals[" II"]
    )
