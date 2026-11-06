from __future__ import annotations

import pytest

from src.t01_loader import load_record


@pytest.fixture
def waveform():
    return load_record("01", data_source="synthetic")


@pytest.fixture
def two_waveforms():
    return [
        load_record("01", data_source="synthetic"),
        load_record("02", data_source="synthetic"),
    ]
