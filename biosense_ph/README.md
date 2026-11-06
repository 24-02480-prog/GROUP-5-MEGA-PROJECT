# BioSense PH

Educational prototype for signal-quality assessment and rate estimation using
public physiological waveform data. It is not a medical device. Any flags
shown by the project are simulation-only demo flags.

The project was developed and tested first with deterministic synthetic
stand-in records because the BIDMC CSV files were not present during
development. Synthetic results are labeled `SYNTHETIC` in logs, report tables,
and figure titles. The synthetic data has no relationship to any individual.

## Setup

Use Python 3.12 and run these commands from the project directory in
PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python run_all.py
streamlit run src/t07_dashboard.py
```

If PowerShell blocks activation for the current terminal, use the venv's
Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe run_all.py
.\.venv\Scripts\streamlit.exe run src/t07_dashboard.py
```

The requirements include Streamlit and the matplotlib dashboard fallback.
No paid service, sensor, or runtime internet access is used.

## Real data

Download the **BIDMC PPG and Respiration Dataset v1.0.0** from
[PhysioNet](https://physionet.org/content/bidmc/1.0.0/), DOI
[10.13026/C2208R](https://doi.org/10.13026/C2208R). Copy these four files into
`data/` without renaming them:

- `bidmc_01_Signals.csv`
- `bidmc_02_Signals.csv`
- `bidmc_01_Numerics.csv`
- `bidmc_02_Numerics.csv`

The loader strips whitespace from the CSV header names. It selects real data
automatically only when all four files are present. Otherwise, `auto` prints
`REAL DATA NOT FOUND - USING SYNTHETIC DATA` and generates two fixed-seed
records. Select a source explicitly with `--data-source real` or
`--data-source synthetic`.

Check the copied files before analysis:

```powershell
python check_real_data.py
```

The checker reports PASS/FAIL for required columns, monotonic timestamps,
125 Hz signal sampling, and signal/Numerics duration coverage. A failed check
exits with a non-zero status.

## Run the project

Run the end-to-end loader, processing, quality, fusion, evaluation, and export
pipeline. The default run includes A/B/C plus each severity scenario listed in
`config.yaml`:

```powershell
python run_all.py
```

To select the source:

```powershell
python run_all.py --data-source auto
python run_all.py --data-source synthetic
python run_all.py --data-source real
```

Run the dashboard after installing its optional requirement:

```powershell
streamlit run src/t07_dashboard.py
```

Without Streamlit, create the static matplotlib dashboard plots and CSVs:

```powershell
python src/t07_dashboard.py --fallback
```

Run the tests:

```powershell
python -m pytest
```

## Outputs

Generated artifacts are written to `outputs/`, including:

- `error_table.csv`: HR/PR/RR MAE, evaluated windows, and valid-window
  coverage before and after quality gating, by record and scenario.
- `final_feature_table.csv`: validated feature rows in the shared schema.
- `fused_timeline.csv`, `fusion_scenario_demos.csv`, and
  `scenario_flag_summary.csv`: fused estimates, reference input, quality, and
  simulation-only demo flags.
- `scenario_quality_pass_fail.csv`, `quality_classifier_metrics.csv`,
  `quality_severity_metrics.csv`, `quality_cv_metrics.csv`, severity/CV
  confusion matrices, `quality_f1_vs_snr.png`, and
  `quality_feature_importance.csv`.
- `ml_rate_correction_metrics.csv`, per-window predictions,
  `peak_based_feature_table.csv`, and `ml_corrected_feature_table.csv`.
- ECG, PPG, respiration, rate-error, confusion-matrix, and feature-importance
  PNG figures, plus per-rate Bland-Altman plots, quality-stratified error, and
  coverage-MAE trade-off charts.
- `scenario_explanation.md`: explanation of A/B/C scenarios and quality gating.

The feature-table columns are `record_id`, `start_s`, `end_s`, `feature`,
`value`, `unit`, and `quality`. Features include HR, PR, RR, provided
SpO2-reference, HR/PR absolute rate difference, quality, and scenario flag.
Windows are non-overlapping 30-second intervals; quality labels are `OK`,
`LOW`, or `MISSING`. Exported feature-table record IDs carry a `synthetic-` or
`real-` prefix so source remains visible without changing the table schema.

## Data lineage and rate processing

The raw signals are ECG (`II`), PPG (`PLETH`), and respiration (`RESP`).
Processed signals use a zero-phase Butterworth band-pass: 0.5-40 Hz for ECG,
0.5-8 Hz for PPG, and 0.1-0.7 Hz for respiration. ECG and PPG peak detection
uses a 0.45-second minimum peak spacing; respiration uses 1.5 seconds.
Windowed HR, PR, and RR are calculated from the first and last detected peak
times after refractory peak detection and median filtering of inter-beat
interval outliers. Welch spectral rate estimates cross-check each peak-based
estimate; disagreements over configured limits lower the rate quality. The
ECG-HR/PPG-PR absolute difference is also checked per window and is labeled
LOW above its configured limit. `SpO2_ref` is read only from the provided
Numerics reference; it is never derived from PPG.

Rule-based quality uses missing fraction, flatline ratio, robust outlier
counts, amplitude spread, and band-power ratio. Machine-learning comparisons
use a RandomForest and LogisticRegression with record-wise holdout in both
directions (record 01 -> 02 and 02 -> 01), and 5-fold stratified
cross-validation is run only within each training record. Severity scenarios
include Gaussian noise at 20, 10, 5, and 0 dB SNR; spike bursts of 5, 20, and
50 spikes per 30-second window; 10%, 30%, and 60% partial dropouts; two levels
of baseline drift; and 50%/20% amplitude retention. The per-window Ridge rate
correction is fitted on one record and tested on the other, in both
directions; peak-based and corrected MAE are reported separately. `config.yaml` contains the new scenario, model, cross-check, and robust-rate
thresholds.
Simulated corruption is marked separately from the raw-data source. Non-OK
rate values are gated to NaN.
`NO-DATA` is used when required quality is unavailable; threshold flags remain
separate and use non-clinical demo labels.

The quality pass/fail report includes a clean control and three independent
modified examples for each signal: noise-only, spikes-only, and
flatline/missing.

## Evaluation requirement and limitations

All metrics currently produced without the BIDMC CSV files are synthetic
development results. **Re-run every reported metric on the real data before
submission.** Do not present synthetic metrics as real-data results.

Known limitations include the two-record scope, synthetic corruption
patterns that do not represent all real signal artifacts, class balance and
correlation in small window-level cross-validation samples, limited training
data for the cross-record rate correction, peak-detector failure cases,
Welch resolution limits in 30-second windows, and thresholds chosen only for
this educational demonstration. ML metrics and corrections must be re-run on
the real data. Results should not be used for personal decision-making.

## References and privacy

- Pimentel et al., PhysioNet, *BIDMC PPG and Respiration Dataset*, v1.0.0,
  DOI [10.13026/C2208R](https://doi.org/10.13026/C2208R).
- Pimentel et al., IEEE Transactions on Biomedical Engineering, 2016,
  DOI [10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124).

The referenced dataset is public and de-identified. Handle downloaded copies
according to the dataset terms and applicable privacy requirements, including
the Philippines Data Privacy Act of 2012 (RA 10173). Do not add identifiable
personal data to this project.
