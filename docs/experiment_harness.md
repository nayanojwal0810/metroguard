# MetroGuard Reproducible Experiment Harness Specification

## 1. Overview and Architecture

The experiment harness provides a reproducible, leakage-safe pipeline to train configurable Sparse Autoencoder (SAE) models on chronological `TRAIN` data and evaluate them on `CALIBRATION` data without touching `FINAL HOLDOUT`.

```
Raw CSV / Streaming Chunks
          │
          ▼
┌──────────────────┐
│ Holdout-Safe     │  --> Reads ONLY timestamp + 7 primary features
│ Ingestion        │  --> Drops pre-TRAIN (< 2020-02-01) and breaks before HOLDOUT (>= 2020-06-01)
└─────────┬────────┘
          │
          ├──> TRAIN Partition (2020-02-01 00:00:00 to 2020-03-31 23:59:59; 445,298 rows)
          │         │
          │         ▼
          │    Chronological Split: TRAIN-fit (e.g. 90%) and TRAIN-validation (e.g. 10%)
          │         │
          │         ├──> TRAIN-fit Causal Windows
          │         │         │
          │         │         ▼
          │         │    Scaler Fit (StandardScaler / MinMaxScaler fit STRICTLY on TRAIN-fit)
          │         │         │
          │         │         ▼
          │         │    TRAIN-fit Transform (using fitted scaler)
          │         │
          │         └──> TRAIN-val Causal Windows
          │                   │
          │                   ▼
          │              TRAIN-val Transform (using frozen TRAIN-fit scaler; no parameter updates)
          │                   │
          │                   ▼
          │              SAE Training & Early Stopping
          │              (Snapshots best epoch validation parameters; restores best state)
          │
          └──> CALIBRATION Partition (2020-04-01 00:00:00 to 2020-05-31 23:59:59; 411,534 rows)
                    │
                    ▼
               CALIBRATION Causal Windows
                    │
                    ▼
               Calibration Transform (using frozen TRAIN-fit scaler; parameters strictly invariant)
                    │
                    ▼
               Calibration Evaluator
               - Explicit Window Accounting
               - Threshold-free Metrics (PR-AUC, ROC-AUC, Distributions, Quantiles)
               - Labeling Convention: Window End Timestamp
                    │
                    ▼
               Run Metadata & Artifact JSON
```

---

## 2. Frozen Partitions and Verified Row Counts

The authoritative chronological partition boundaries and verified counts are:

| Partition | Start Timestamp | End Timestamp | Verified Rows |
|:---|:---|:---|:---|
| `TRAIN` | `2020-02-01 00:00:00` | `2020-03-31 23:59:59` | 445,298 |
| `CALIBRATION` | `2020-04-01 00:00:00` | `2020-05-31 23:59:59` | 411,534 |
| `FINAL HOLDOUT` | `2020-06-01 00:00:00` | `2020-08-31 23:59:59` | 659,586 |
| `UNUSED TAIL` | `2020-09-01 00:00:00` | `2020-09-01 03:59:50` | 530 |
| **Total** | | | **1,516,948** |

### Strict Holdout Ingestion Guarantee
- Chunked streaming reads only the columns required by the experiment (`timestamp` + 7 primary analogue features).
- Ingestion halts early as soon as the chronological stream passes `CALIBRATION_END` (`2020-05-31 23:59:59`).
- No `FINAL HOLDOUT` observations are retained in or passed through the experiment pipeline.

---

## 3. Train / Validation / Scaler Separation

When a validation split is enabled (`validation_fraction > 0.0`):
1. **Chronological Partitioning**: `train_df` is partitioned chronologically into `train_fit_df` and `train_val_df`.
2. **Strict Fitting**: The scaler is fitted **strictly** on the `train_fit` subset. Observations in `train_val` never enter scaler parameter estimation.
3. **Frozen Invariance**:
   - `train_fit` is transformed with the fitted scaler.
   - `train_val` is transformed with the same frozen scaler.
   - `cal` is transformed with the same frozen scaler.
   - Explicit parameter assertions guarantee that scaler parameters (`mean_`, `std_`, `min_`, `max_`) do not mutate during validation or calibration transforms.

---

## 4. Early Stopping and State Restoration

When validation is enabled in [`SAETrainer`](file:///d:/metroguard/src/experiments/trainer.py):
- The trainer tracks the lowest validation loss.
- Whenever validation loss improves, a detached snapshot of model parameters is saved.
- At the conclusion of training, the model is restored to the parameter state of `best_epoch`.
- If no validation loader is provided, the model state is left as-is without manufacturing an artificial best epoch.

---

## 5. Explicit Window Accounting

Window accounting preserves the distinction between raw observations and sliding window endpoints:
- **`num_observations`**: Total raw rows in the partition.
- **`num_candidate_endpoints`**: Total candidate evaluation timestamps considered with stride $S$.
- **`num_usable_windows`**: Endpoints with a complete, gap-free history of $W$ observations.
- **`num_excluded_insufficient_history`**: Endpoints at the beginning of the series ($i < W - 1$) with fewer than $W$ prior observations.
- **`num_excluded_service_gap`**: Endpoints where a telemetry gap ($\Delta t > 60\text{s}$) prevents forming a continuous causal window.
- **`num_excluded_split_boundary`**: Endpoints where a candidate window crosses partition bounds.
- **`num_excluded_out_of_bounds`**: Endpoints falling outside defined split bounds.

**Conservation Rule**:
$$\text{num\_candidate\_endpoints} = \text{num\_usable\_windows} + \sum \text{num\_excluded\_*}$$

---

## 6. Calibration Labeling Convention

A window is labelled strictly according to its **evaluation/end timestamp** $t$:
- If $t \in [\text{event.start}, \text{event.end}]$: label = 1 (failure), `event_id` = documented event ID.
- If $t < \text{event.start}$ (immediately before event): label = 0 (normal).
- If $t > \text{event.end}$ (immediately after event): label = 0 (normal).
The window's historical context $[t - W + 1, t - 1]$ does not alter event membership.

Documented Calibration Events:
- `Event_1`: `2020-04-18 00:00:00` to `2020-04-18 23:59:59`
- `Event_2`: `2020-05-29 23:30:00` to `2020-05-30 06:00:00`

---

## 7. Dataset Provenance

- **Real MetroPT-3 Runs**: `dataset_fingerprint = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"`
- **Synthetic / Smoke Runs**: `dataset_fingerprint = "synthetic_experiment_dataset"`

---

## 8. Usage Commands

```bash
# Run lightweight synthetic smoke test (< 5 seconds on CPU)
python -m src.experiments.runner --smoke

# Inspect the 16 candidate configurations
python -m src.experiments.runner --list-matrix

# View estimated compute and memory requirements
python -m src.experiments.runner --estimate-resources
```
