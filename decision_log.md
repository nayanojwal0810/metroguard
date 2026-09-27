# MetroGuard — Decision Log

This file records approved technical decisions that affect scientific meaning, evaluation, architecture, or reproducibility.

Do not silently rewrite historical decisions.

## Decision format

### DEC-XXX — [Decision title]

**Date:**  
**Status:** Proposed / Approved / Superseded

**Decision**

[What was decided.]

**Reason**

[Why it was chosen.]

**Evidence**

[Experiment, source, metric, artifact, or assumption.]

**Alternatives**

[Important alternatives considered.]

**Impact**

[What this decision changes.]

**Approved by**

ChatGPT / User

---

## Initial decisions

### DEC-001 — Project objective

**Status:** Approved

**Decision**

MetroGuard focuses on turning compressor anomaly scores into a defensible maintenance-alert decision.

**Reason**

This preserves the original anomaly-detection formulation while creating a focused engineering contribution.

**Evidence**

Approved project execution specification.

**Impact**

New neural architectures are outside the current core scope unless later approved.

**Approved by**

ChatGPT / User

### DEC-002 — Primary dataset

**Status:** Approved

**Decision**

Use the UCI MetroPT-3 dataset as the primary dataset.

**Reason**

It provides real compressor telemetry and documented failure intervals suitable for anomaly-detection evaluation.

**Impact**

Dataset version and hash must be recorded before model evaluation.

**Approved by**

ChatGPT / User

### DEC-003 — Baseline formulation

**Status:** Approved

**Decision**

Use a sparse autoencoder as the primary research baseline, reproduced from the selected source as faithfully as practical.

**Impact**

Baseline assumptions must be documented before modifications are introduced.

**Approved by**

ChatGPT / User

### DEC-004 — Protected evaluation

**Status:** Approved

**Decision**

Use chronological training → calibration → final holdout separation.

**Impact**

The final holdout cannot be used for tuning.

**Approved by**

ChatGPT / User

### DEC-005 — MetroGuard modifications

**Status:** Approved

**Decision**

Evaluate:
- cost-aware alert policy;
- operating-regime-aware policy;
- input-quality gate.

**Impact**

These components must be tested individually and together through controlled ablations.

**Approved by**

ChatGPT / User

### DEC-006 — CPU/Colab integration

**Status:** Approved

**Decision**

Git is the synchronization boundary. One canonical codebase is shared across CPU and Colab.

**Impact**

No independent Colab source tree is permitted.

**Approved by**

ChatGPT / User

### DEC-007 — Dataset integrity and empirical sampling cadence

**Status:** Approved

**Decision**

Adopt `data/raw/MetroPT3(AirCompressor).csv` as the verified dataset artifact for MetroGuard with SHA-256 hash `DB30CCB4EA402E3C8BF2C99DB06E288D4F2A772F6928F9DBE26A920D69793E24`, 1,516,948 observations, and recognize its nominal sampling cadence as ~10 seconds (~0.1 Hz) rather than 1 Hz.

**Reason**

Direct empirical analysis of timestamps proves 88.17% of observations are spaced at 10.0s intervals, resolving conflicting 1 Hz wording in the published metadata.

**Evidence**

- SHA-256 verification and row count in `docs/dataset_notes.md`.
- Timestamp difference analysis in `docs/dataset_notes.md`.

**Alternatives**

Assuming 1 Hz sampling cadence (which would distort all physical rate calculations, feature windows, and duration estimates by a factor of 10).

**Impact**

All temporal windowing, lead-time calculations, and feature engineering will explicitly assume nominal 10-second intervals and handle physical service gaps causally.

**Approved by**

ChatGPT / User

### DEC-008 — Primary baseline input feature representation

**Status:** Approved

**Decision**

Seven analogue sensors are the primary MetroGuard baseline input: `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, and `Motor_current`.

**Reason**

The project focuses on documented compressor air-leak events and requires a stable, interpretable baseline input representation. The 15-sensor configuration (incorporating the 8 digital signals) is reserved for a future controlled feature ablation.

**Evidence**

Domain relevance to air-leak dynamics and baseline design specification in `docs/baseline_design.md`.

**Alternatives**

- All 15 sensor channels (reserved for ablation).
- Bivariate `TP3 + Motor Current` (rejected as unsupported secondary simplification).

**Impact**

The initial baseline sparse autoencoder and causal window preprocessing will ingest the 7 analogue sensors. This is a MetroGuard engineering decision, not an exact claim from external literature. The window size remains open among candidate values.

**Approved by**

ChatGPT / User

### DEC-009 — Chronological evaluation boundaries and leakage-safe data contract

**Status:** Approved

**Decision**

Partition the MetroPT-3 dataset strictly along non-overlapping chronological boundaries:
- **TRAIN:** `2020-02-01 00:00:00` through `2020-03-31 23:59:59` (445,298 rows, 0 failure events);
- **CALIBRATION:** `2020-04-01 00:00:00` through `2020-05-31 23:59:59` (411,534 rows, 2 failure events: April 18, May 29–30);
- **FINAL HOLDOUT:** `2020-06-01 00:00:00` through `2020-08-31 23:59:59` (659,586 rows, 2 failure events: June 5–7, July 15);
- **UNUSED PARTIAL TAIL:** `2020-09-01 00:00:00` through `2020-09-01 03:59:50` (530 rows, excluded from primary evaluation).

Preprocessing scalers are fitted exclusively on `TRAIN`. Windows cannot cross split boundaries or service gaps ($\Delta t > 60\text{s}$).

**Reason**

Strict prevention of temporal data leakage. Guarantees a completely healthy 2-month baseline period for unsupervised model fitting, a 2-month realistic calibration period with ground-truth failure events for tuning window size and alert thresholds, and a protected 3-month holdout for forward-looking operational evaluation. The September 1 tail represents an incomplete partial day without a full diurnal cycle and is safely excluded from primary evaluation while remaining in the raw CSV.

**Evidence**

Dataset timestamp verification and failure interval audit documented in `docs/data_contract.md` and validated in unit tests.

**Alternatives**

- Random train/test split (strictly rejected as scientifically invalid and prone to extreme leakage).
- Single train/test split without an intermediate calibration partition (rejected because threshold and window selection would either contaminate training or leak the holdout).

**Impact**

Data ingestion, scalers, and causal window construction enforce split boundaries with zero temporal leakage. Final holdout is strictly protected from model and threshold tuning.

**Approved by**

ChatGPT / User

---

## Reserved decisions

Do not fill these from assumption. Record them only after evidence is available.

- window size winner (from candidate set {6, 30, 90, 180});
- scaling method winner (StandardScaler vs MinMaxScaler);
- baseline architecture details (hidden layers, latent dimension);
- sparsity mechanism (L1 vs KL) and regularization weight $\lambda$;
- baseline threshold selection method and value;
- alert episode semantics (persistence $K$, smoothing, cooldown);
- cost objective and weights;
- regime definition;
- quality-gate thresholds;
- monitoring thresholds;
- retraining triggers;
- serving/tracking stack.
