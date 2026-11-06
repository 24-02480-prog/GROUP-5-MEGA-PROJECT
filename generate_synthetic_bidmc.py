"""
Generate synthetic, BIDMC-format physiological recordings (no personal data).

Creates, for each synthetic record NN:
  synthetic_NN_Signals.csv   125 Hz: " Time [s]", " RESP", " PLETH", " V", " AVR", " II"
  synthetic_NN_Numerics.csv  1 Hz:   " Time [s]", " HR", " PULSE", " RESP", " SpO2"

Column names keep the leading spaces used by the real PhysioNet BIDMC CSVs.
Reference values (HR/PULSE/RESP) are computed from the same underlying beat
and breath times used to build the waveforms, so error evaluation is meaningful.

Educational use only. These are simulated signals, not clinical data.

Usage:  python generate_synthetic_bidmc.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FS = 125          # Hz, same as BIDMC
DURATION = 480    # s (8 minutes, same as BIDMC)


def _gauss(t, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def make_record(seed: int, mean_hr: float, mean_rr: float, spo2_base: float):
    rng = np.random.default_rng(seed)
    n = FS * DURATION
    t = np.arange(n) / FS

    # --- respiration: phase-integrated so rate varies slowly ---
    rr_inst = mean_rr + 2.0 * np.sin(2 * np.pi * t / 170.0 + rng.uniform(0, 6.28)) \
        + 0.8 * np.sin(2 * np.pi * t / 53.0 + rng.uniform(0, 6.28))   # breaths/min
    rr_inst = np.clip(rr_inst, 8, 28)
    resp_phase = 2 * np.pi * np.cumsum(rr_inst / 60.0) / FS
    resp = np.sin(resp_phase)
    resp_breath_times = t[np.where(np.diff(np.floor(resp_phase / (2 * np.pi))) > 0)[0]]

    # --- heart beats: HR varies slowly + respiratory sinus arrhythmia ---
    beat_times = []
    cur = 0.4
    while cur < DURATION - 0.5:
        i = min(int(cur * FS), n - 1)
        hr = mean_hr + 6.0 * np.sin(2 * np.pi * cur / 140.0) \
            + 2.5 * resp[i] + rng.normal(0, 0.6)
        hr = float(np.clip(hr, 45, 140))
        beat_times.append(cur)
        cur += 60.0 / hr
    beat_times = np.array(beat_times)

    # --- ECG (lead II like): P, Q, R, S, T as gaussians per beat ---
    ecg = np.zeros(n)
    for bt in beat_times:
        lo, hi = int(max(0, (bt - 0.4) * FS)), int(min(n, (bt + 0.6) * FS))
        tt = t[lo:hi]
        ecg[lo:hi] += (_gauss(tt, bt - 0.16, 0.025, 0.12)    # P
                       + _gauss(tt, bt - 0.025, 0.010, -0.15)  # Q
                       + _gauss(tt, bt, 0.012, 1.0)          # R
                       + _gauss(tt, bt + 0.028, 0.012, -0.25)  # S
                       + _gauss(tt, bt + 0.25, 0.045, 0.30))   # T
    ecg += 0.08 * resp + 0.05 * np.sin(2 * np.pi * 0.07 * t)   # baseline wander
    ecg += rng.normal(0, 0.015, n)

    # --- PPG: systolic peak + dicrotic bump, delayed after R, resp-modulated ---
    ppg = np.zeros(n)
    for bt in beat_times:
        pt = bt + 0.22  # pulse transit delay
        lo, hi = int(max(0, (pt - 0.3) * FS)), int(min(n, (pt + 0.8) * FS))
        tt = t[lo:hi]
        i = min(int(pt * FS), n - 1)
        amp = 1.0 + 0.15 * resp[i]
        ppg[lo:hi] += amp * (_gauss(tt, pt, 0.09, 1.0) + _gauss(tt, pt + 0.28, 0.10, 0.35))
    ppg += 0.10 * resp + 0.03 * np.sin(2 * np.pi * 0.05 * t)
    ppg += rng.normal(0, 0.01, n)

    # --- other BIDMC columns (kept for format compatibility) ---
    v = ecg * 0.6 + rng.normal(0, 0.01, n)
    avr = -0.5 * ecg + rng.normal(0, 0.01, n)

    signals = pd.DataFrame({
        " Time [s]": np.round(t, 3),
        " RESP": np.round(resp, 5),
        " PLETH": np.round(ppg, 5),
        " V": np.round(v, 5),
        " AVR": np.round(avr, 5),
        " II": np.round(ecg, 5),
    })

    # --- numerics at 1 Hz, from the true underlying events ---
    sec = np.arange(DURATION)
    hr_ref, rr_ref = [], []
    for s in sec:
        w_b = beat_times[(beat_times >= s - 4) & (beat_times < s + 4)]
        hr_ref.append(60.0 * (len(w_b) - 1) / (w_b[-1] - w_b[0]) if len(w_b) > 2 else np.nan)
        w_r = resp_breath_times[(resp_breath_times >= s - 15) & (resp_breath_times < s + 15)]
        rr_ref.append(60.0 * (len(w_r) - 1) / (w_r[-1] - w_r[0]) if len(w_r) > 2 else np.nan)
    hr_ref = pd.Series(hr_ref).bfill().ffill().to_numpy()
    rr_ref = pd.Series(rr_ref).bfill().ffill().to_numpy()
    spo2 = np.clip(spo2_base + np.cumsum(rng.normal(0, 0.08, DURATION)) * 0.3, 90, 100)

    numerics = pd.DataFrame({
        " Time [s]": sec,
        " HR": np.round(hr_ref, 1),
        " PULSE": np.round(hr_ref + rng.normal(0, 0.3, DURATION), 1),
        " RESP": np.round(rr_ref, 1),
        " SpO2": np.round(spo2, 1),
    })
    return signals, numerics


def main(out_dir: str = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {1: (74.0, 15.0, 97.0), 2: (88.0, 19.0, 95.0)}
    for k, (hr, rr, sp) in specs.items():
        sig, num = make_record(seed=100 + k, mean_hr=hr, mean_rr=rr, spo2_base=sp)
        sig.to_csv(out / f"synthetic_{k:02d}_Signals.csv", index=False)
        num.to_csv(out / f"synthetic_{k:02d}_Numerics.csv", index=False)
        print(f"synthetic_{k:02d}: {len(sig)} samples @ {FS} Hz, "
              f"ref HR {num[' HR'].min():.0f}-{num[' HR'].max():.0f} bpm, "
              f"RR {num[' RESP'].min():.0f}-{num[' RESP'].max():.0f} /min")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
