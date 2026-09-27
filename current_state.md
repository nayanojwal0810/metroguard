# MetroGuard — Current State

## Active objective

Configurable Sparse Autoencoder baseline implementation and loss module verified; preparation for controlled baseline training experiments.

## Current status

Baseline model layer established:
- Data foundation audit complete: raw dataset SHA-256 confirmed (`db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24`); cadence descriptions corrected across all docs to audited values (predominantly 9–13s, 36 sub-60s intervals, 331 service gaps > 60s); $W$ explicitly defined as observation count.
- Scaler enforcement active: `BaseScaler` programmatically blocks fitting on non-TRAIN partitions and non-TRAIN timestamps.
- Model design published: `docs/sae_design.md` specifies symmetric MLP architecture ($D \to H \to Z \to H \to D$), canonical dimensionality rule, and L1/KL sparsity formulations.
- PyTorch implementation completed: `src/models/sae.py`, `src/models/loss.py`, and `src/models/__init__.py`.
- Unit test suite expanded: 27 unit tests passing across `tests/`, verifying model shapes (2D/3D), deterministic inference, sample-level reconstruction errors, L1/KL sparsity penalties, and dimension validation.
- Model training has NOT started.
- Window size winner, scaler winner, sparsity mechanism winner, threshold, and alert semantics remain open technical decisions.

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
