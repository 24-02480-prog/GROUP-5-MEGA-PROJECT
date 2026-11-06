from __future__ import annotations

import numpy as np

from src import t01_loader
from src.synthetic_data import generate_synthetic_record


def test_real_loader_strips_header_spaces(tmp_path, monkeypatch, capsys):
    source = generate_synthetic_record(1)
    source.signals.to_csv(tmp_path / "bidmc_01_Signals.csv", index=False)
    source.numerics.to_csv(tmp_path / "bidmc_01_Numerics.csv", index=False)
    monkeypatch.setattr(t01_loader, "DATA_DIR", tmp_path)
    waveform = t01_loader.load_record("01", "real")
    assert waveform.data_source == "REAL"
    assert np.isclose(waveform.fs, 125.0)
    assert set(waveform.signals) == {"RESP", "PLETH", "V", "AVR", "II"}
    assert "REAL" in capsys.readouterr().out


def test_loader_falls_back_to_synthetic_when_files_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(t01_loader, "DATA_DIR", tmp_path)
    waveform = t01_loader.load_record("02", "auto")
    assert waveform.data_source == "SYNTHETIC"
    assert "REAL DATA NOT FOUND - USING SYNTHETIC DATA" in capsys.readouterr().out


def test_load_records_produces_two_source_labeled_records(tmp_path, monkeypatch):
    monkeypatch.setattr(t01_loader, "DATA_DIR", tmp_path)
    records = t01_loader.load_records("synthetic")
    assert [record.data_source for record in records] == ["SYNTHETIC", "SYNTHETIC"]
    assert all(len(record.time_s) == 60_000 for record in records)
