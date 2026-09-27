# MetroGuard — Data Contract & Evaluation Protocol

## 1. Purpose

This data contract establishes the authoritative technical specifications, boundaries, and leakage constraints governing all data ingestion, temporal windowing, preprocessing, and model evaluation in MetroGuard.

## 2. Dataset Identity

| Attribute | Specification |
|---|---|
| **Canonical File Path** | `data/raw/MetroPT3(AirCompressor).csv` |
| **Row Count** | `1,516,948` rows |
| **Column Count** | `17` columns (1 index, 1 timestamp, 15 sensor features) |
| **SHA-256 Checksum** | `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24` |
| **Chronological Start** | `2020-02-01 00:00:00` |
| **Chronological End** | `2020-09-01 03:59:50` |
| **Sampling Cadence** | Empirical ~10-second nominal cadence ($99.976\%$ within 9–13s range; $0.022\%$ service breaks $>60\text{s}$) |
| **Missing Values** | `0` null / missing entries across all columns |
| **Duplicate Timestamps** | `0` duplicates; timestamps strictly monotonically non-decreasing |

## 3. Feature Contract

### Primary Baseline Features

The approved primary baseline input (**METROGUARD ENGINEERING DECISION**, `DEC-008`) consists of the seven continuous analogue sensor signals present in the local MetroPT-3 dataset:

1. `TP2` — Pressure on compressor (bar)
2. `TP3` — Pneumatic panel pressure (bar)
3. `H1` — Cyclonic separator filter pressure drop (bar)
4. `DV_pressure` — Air dryer discharge pressure drop (bar)
5. `Reservoirs` — Downstream reservoir tank pressure (bar)
6. `Oil_temperature` — Compressor oil temperature (°C)
7. `Motor_current` — Compressor motor electric current (A)

### Excluded / Retained Columns

- **Serialized Index:** The first column (`Unnamed: 0` in pandas / `column00`) is an auto-generated integer index from CSV export and must be omitted from all feature matrices.
- **Timestamp Column:** `timestamp` is used strictly for causal ordering, gap detection, and split assignment. It is never fed as a numeric feature into the model.
- **Digital Status Signals (8):** `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` are retained in the raw dataset for system context and future controlled feature ablations, but are strictly excluded from primary baseline input.

## 4. Chronological Split Contract

The dataset is partitioned strictly along chronological boundaries to prevent temporal leakage and evaluate model performance under realistic forward-looking operational conditions:

| Partition | Start Timestamp | End Timestamp | Row Count | % of Dataset | Ground-Truth Failure Events |
|---|---|---|---|---|---|
| **TRAIN** | `2020-02-01 00:00:00` | `2020-03-31 23:59:59` | 445,298 | 29.355% | 0 failures (healthy baseline operations) |
| **CALIBRATION** | `2020-04-01 00:00:00` | `2020-05-31 23:59:59` | 411,534 | 27.129% | 2 air leak failures (April 18, May 29–30) |
| **FINAL HOLDOUT** | `2020-06-01 00:00:00` | `2020-08-31 23:59:59` | 659,586 | 43.481% | 2 air leak failures (June 5–7, July 15) |
| **UNUSED TAIL** | `2020-09-01 00:00:00` | `2020-09-01 03:59:50` | 530 | 0.035% | Excluded from primary evaluation |
| **Total** | `2020-02-01 00:00:00` | `2020-09-01 03:59:50` | 1,516,948 | 100.000% | 4 ground-truth failure intervals |

### Split Rules

1. **Strictly Chronological:** Partitions are non-overlapping in time: $\text{Train} < \text{Calibration} < \text{Final Holdout} < \text{Unused Tail}$. Random shuffling or cross-validation is strictly prohibited.
2. **Train Partition Purpose:** Used solely to fit preprocessing transformations (scalers) and train the unsupervised Sparse Autoencoder to learn healthy compressor dynamics.
3. **Calibration Partition Purpose:** Used to select temporal window size $W$, tune anomaly threshold $\tau$, calibrate alert persistence $K$, and evaluate false alarm rates on operational data containing real failure events.
4. **Final Holdout Partition Purpose:** Protected out-of-sample test partition for final forward-looking operational metric reporting. Access is strictly forbidden during model tuning or hyperparameter selection.
5. **Partial-Tail Treatment:** The 530 rows on September 1, 2020 represent an incomplete 4-hour tail recorded prior to telemetry termination. Because it spans only a partial morning fraction without a full diurnal cycle and contains no failure events, it is excluded from the primary evaluation holdout to maintain clean calendar month boundaries (June 1 – August 31). It remains preserved in the raw CSV.

## 5. Leakage Contract

To guarantee that offline evaluation reflects real-world operational deployment on high-speed rail systems, the following core invariant is enforced across the entire codebase:

$$\forall t, \quad \text{Decision}(t) = f\big(\{x_\tau \mid \tau \le t\}\big)$$

**Only information timestamped at or before time $t$ may influence an inference or maintenance decision at time $t$.**

### Explicit Prohibitions

1. **Future-Row Scaling:** Normalization parameters (mean, standard deviation, min, max) must be computed exclusively on the `TRAIN` partition. Scalers must never be fitted on `CALIBRATION` or `HOLDOUT`, nor across the full dataset.
2. **Future Interpolation:** Missing data or gaps must never be filled using forward interpolation, spline fitting, or bidirectional smoothing.
3. **Centered Rolling Windows:** Rolling statistics or windowing must strictly look backward into the past: $[t_{i - W + 1}, \, t_i]$. Centered windows $[t - k, \, t + k]$ are strictly prohibited.
4. **Future-Derived Imputation:** Any missing value strategy using global statistics calculated over future rows is prohibited.
5. **Post-Event Feature Construction:** Features derived from knowing when an alert or maintenance event terminated are strictly prohibited.
6. **Holdout-Driven Tuning:** No hyperparameter, threshold, architecture parameter, or window candidate may be selected based on holdout performance.
7. **Windows Crossing Split Boundaries:** A temporal window evaluated in `CALIBRATION` must not ingest rows from `TRAIN`; a temporal window in `FINAL HOLDOUT` must not ingest rows from `CALIBRATION` or `TRAIN`.
8. **Windows Crossing Service Gaps:** A temporal window must not bridge across operational breaks where $\Delta t > 60\text{ seconds}$.

## 6. Gap Contract

The local MetroPT-3 telemetry exhibits 331 physical service gaps where consecutive timestamp differences exceed 60 seconds (up to 48.03 hours).

- **Gap Reset Rule:** If $\Delta t_i = t_i - t_{i-1} > 60\text{ seconds}$, a service break is declared.
- **Engineering Status:** The 60-second boundary is an empirical MetroGuard engineering decision, reflecting operational compressor cycling and telemetry intermissions, not a literature claim.
- **Insufficient History Handling:** Following a service break or partition boundary, until $W$ consecutive continuous observations accumulate without a gap, no model-ready window is produced. The pipeline must explicitly mark the state as `INSUFFICIENT_HISTORY` or `WARMUP` rather than fabricating padded inputs or bridging across the break.

## 7. Authoritative Windowing Contract

The authoritative definition of a temporal window in MetroGuard is:

- A window contains exactly $W$ consecutive observations.
- The window ends at the evaluation observation $t$.
- All observations are at or before $t$.
- Every consecutive timestamp gap within the window must be $\le 60\text{ seconds}$.
- A window may not cross a train/calibration/holdout boundary.
- A window may not cross a service gap.
- No padding, interpolation, or synthetic observations are introduced.
- **Therefore, $W$ represents observation count, not an exact elapsed-time duration.**

The cadence is predominantly 9–13 seconds (99.976% of intervals), a small number of larger sub-60-second intervals exist (36 intervals between 13s and 60s), and 331 service gaps have $\Delta t > 60\text{ seconds}$. Consequently, physical elapsed time across $W$ observations varies naturally; $W$ represents observation count, not an elapsed-time duration, and must never be equated with an exact elapsed-time interval.
