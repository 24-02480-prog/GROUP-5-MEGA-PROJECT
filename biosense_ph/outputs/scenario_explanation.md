# Scenario demonstrations and quality gating

All current examples are generated stand-in data. Their records and output
tables are labeled `SYNTHETIC`; these examples are not measurements from
PhysioNet or from a person.

- **Scenario A - unmodified:** estimated HR, PR, and RR use the raw signals
  after band-pass processing and peak detection. Values with an `OK` quality
  label remain available to the fusion step.
- **Scenario B - noise and spikes:** deterministic corruption is added to the
  signal samples. The rule-based checks can label affected windows `LOW`.
- **Scenario C - flatline and missing samples:** deterministic flatline and
  missing regions are introduced. Windows with enough missing samples are
  labeled `MISSING`.

Fusion gates each rate by its quality label. A non-`OK` rate is set to NaN;
windows with any non-`OK` quality produce `NO-DATA`, not a threshold flag.
The demonstration flags are simulation-only and use the labels
`NORMAL-DEMO`, `CAUTION-DEMO`, and `NO-DATA`. A separate threshold-override
example triggers `CAUTION-DEMO` without changing the waveform.

HR is a calculated feature from the processed ECG (`II`), PR is a calculated
feature from the processed PPG (`PLETH`), and RR is a calculated feature from
the processed respiration signal (`RESP`). SpO2 is read only from the provided
Numerics reference column; it is not calculated from PPG. Scenario corruption
is explicitly marked as simulated. These outputs are an educational prototype
and are not intended for personal decision-making.

The quality pass/fail export includes a clean control plus three independent
modified examples per signal: noise-only, spikes-only, and flatline/missing.

The expanded scenario IDs in `config.yaml` add four Gaussian-noise SNR levels
(20, 10, 5, and 0 dB), spike counts of 5/20/50 per 30-second window,
10%/30%/60% dropouts, low/high baseline drift, and 50%/20% amplitude
retention. They are deterministic and remain simulation-only. Severity-level
classifier results use record-wise holdout; five-fold stratified validation is
performed only within the training record.

ECG/PPG and respiration peak-based rate estimates now use refractory periods,
median-filtered interval outlier correction, and Welch spectral cross-checks.
The per-window HR/PR difference is included in the agreed feature-table schema
and receives `LOW` quality above its configured threshold. Additional
validation exports include per-rate Bland-Altman plots, error grouped by
quality label, and coverage versus MAE with quality gating on/off.

The separate Ridge rate-correction outputs identify `peak-based` and
`ML-corrected` values explicitly. Each correction model is fitted on one
record and evaluated on the other; the test record is not used for tuning.
