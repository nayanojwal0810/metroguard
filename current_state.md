# MetroGuard — Current State

## Active objective

Baseline Sparse Autoencoder architecture and training pipeline specification under leakage-safe data contract.

## Current status

Leakage-safe data foundation established:
- Chronological evaluation boundaries approved (`DEC-009`):
  - `TRAIN`: 2020-02-01 to 2020-03-31 (445,298 rows, healthy baseline)
  - `CALIBRATION`: 2020-04-01 to 2020-05-31 (411,534 rows, 2 failure events)
  - `FINAL HOLDOUT`: 2020-06-01 to 2020-08-31 (659,586 rows, 2 failure events, protected)
  - `UNUSED TAIL`: 2020-09-01 (530 rows, excluded from primary evaluation)
- Data contract created: `docs/data_contract.md` establishes schema, primary 7 analogue features, gap rules (Δt > 60s), and strict anti-leakage invariants.
- Baseline design updated: `docs/baseline_design.md` incorporates full preprocessing flow, candidate scalers (StandardScaler, MinMaxScaler), and causal window rules.
- Data ingestion and validation implemented: `src/data/contract.py`, `src/data/validator.py`, `src/data/windowing.py`, and `src/preprocessing/scalers.py`.
- Unit test suite implemented: 16 unit tests passing across `tests/`, verifying schema checks, non-overlapping splits, gap resets, causal windowing, fit/transform separation, and anti-leakage invariant (future row cannot enter past window).
- Model implementation has NOT started.
- Model architecture, window winner, scaler winner, threshold, and alert semantics remain open.

## Approved project direction

- MetroPT-3 is the primary dataset.
- Sparse autoencoder is the primary baseline formulation.
- MetroGuard focuses on anomaly-to-maintenance decisioning.
- Cost-aware alerting, regime-aware policy, and input-quality gating are the approved modifications.
- Chronological leakage-safe evaluation is mandatory.
- Final holdout remains protected from tuning.
- CPU and Colab share one Git-controlled codebase.

## Not yet frozen (Open Technical Decisions)

- exact window size winner (from candidate set {6, 30, 90, 180});
- scaling method winner (StandardScaler vs MinMaxScaler);
- SAE architecture (layer count, hidden unit dimensions, bottleneck size);
- sparsity mechanism (L1 activity regularization vs KL divergence) and weight λ;
- training configuration (optimizer, learning rate, batch size, epochs);
- threshold selection method and value;
- alert semantics (persistence K, smoothing, cooldown);
- cost function weights;
- operating regime definition;
- quality gating thresholds.

## Current work boundary

Allowed:

- repository inspection;
- control-document review;
- dataset acquisition and integrity verification;
- research-source verification;
- lightweight environment checks.

Not allowed yet:

- final model training;
- holdout tuning;
- production implementation;
- scope expansion;
- unapproved model changes.

## Handoff rule

Antigravity must read this file before starting a new task.

After meaningful work, update this file only through the approved workflow so that it reflects the actual repository state.

## Current decision gate

Before model implementation, the project must have:

- verified dataset identity;
- documented baseline assumptions;
- a reproducible chronological data boundary;
- leakage checks planned;
- an approved first implementation objective.
