# MetroGuard — Current State

## Active objective

Baseline input representation decision and decision-gate approval before preprocessing implementation.

## Current status

Baseline design specification phase active:
- Research and data audit complete: primary literature reviewed as background reference; unsupported secondary claims rejected from baseline definition.
- Dataset integrity verified: `data/raw/MetroPT3(AirCompressor).csv` empirically verified (1,516,948 rows, 0 nulls, 0 duplicates, 4 ground-truth failure intervals).
- Cadence verified: complete distribution across all 1,516,947 intervals reconciled (99.976% in 9–13s range; empirical ~10s nominal cadence).
- Primary baseline feature direction approved (METROGUARD ENGINEERING DECISION): 7 continuous analogue sensors (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`). 15-sensor representation reserved for future ablation.
- Temporal window candidate set defined: candidate set {6, 30, 90, 180} observations (~1m, ~5m, ~15m, ~30m) at nominal ~10s cadence under provisional causal gap-reset rule (Δt > 60s).
- Baseline design document created: `docs/baseline_design.md`.
- Model implementation has NOT started.
- Exact window size, scaling method, model architecture, sparsity formulation, training parameters, threshold, and alert semantics remain open technical decisions.

## Approved project direction

- MetroPT-3 is the primary dataset.
- Sparse autoencoder is the primary baseline formulation.
- MetroGuard focuses on anomaly-to-maintenance decisioning.
- Cost-aware alerting, regime-aware policy, and input-quality gating are the approved modifications.
- Chronological leakage-safe evaluation is mandatory.
- Final holdout remains protected from tuning.
- CPU and Colab share one Git-controlled codebase.

## Not yet frozen (Open Technical Decisions)

- exact window size (from candidate set {6, 30, 90, 180});
- scaling method and clipping boundaries;
- SAE architecture (layer count, hidden unit dimensions, bottleneck size);
- sparsity mechanism (L1 activity regularization vs KL divergence) and weight λ;
- training configuration (optimizer, learning rate, batch size, epochs);
- threshold selection method and value;
- alert semantics (persistence K, smoothing, cooldown);
- chronological train/calibration/holdout split dates;
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
