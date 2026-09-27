# MetroGuard — Current State

## Active objective

Preparation and execution of controlled baseline training experiments.

## Current status

Baseline model implementation complete; awaiting training experiment execution:
- SAE implementation is complete as a configurable baseline in PyTorch (`src/models/sae.py`, `src/models/loss.py`).
- Leakage-safe data foundation is implemented and enforced (`src/data/`, `src/preprocessing/`).
- 27 unit tests are currently passing across data validation, chronological splitting, causal windowing, train-only scaler enforcement, and SAE mechanics.
- Model training and controlled baseline experiments have **NOT** started.
- The next approved objective is the preparation and execution of controlled baseline training experiments.
- Final holdout (`2020-06-01` to `2020-08-31`) remains strictly protected from training, inspection, and hyperparameter tuning.
- Calibration partition (`2020-04-01` to `2020-05-31`) is the only partition permitted for model/configuration selection.
- No production decision-layer implementation (cost-aware alerting, regime determination, input-quality gating) should begin yet.

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
- SAE architecture configuration winner (hidden/latent dimensions);
- sparsity mechanism winner (L1 activity regularization vs KL divergence) and weight λ;
- training configuration (optimizer, learning rate, batch size, epoch budget, early stopping);
- anomaly threshold selection method and value (from calibration partition only);
- alert semantics (persistence K, smoothing, cooldown);
- cost function weights;
- operating regime definition;
- quality gating thresholds.

## Current work boundary

Allowed:

- repository inspection and documentation synchronization;
- dataset integrity verification;
- baseline experiment configuration design and runner scripts;
- training model candidates exclusively on the TRAIN partition;
- evaluating candidate windows and thresholds exclusively on the CALIBRATION partition;
- lightweight CPU smoke tests and environment checks.

Not allowed yet:

- inspecting, evaluating on, or tuning against the protected FINAL HOLDOUT;
- implementing production decision-layer modifications (regime policy, cost policy, quality gate);
- claiming model performance or anomaly detection efficacy before experimental verification;
- scope expansion or unapproved architecture alterations.

## Handoff rule

Antigravity must read this file before starting a new task.

After meaningful work, update this file only through the approved workflow so that it reflects the actual repository state.

## Current decision gate

Before controlled baseline experiments, the project must have:

- verified dataset identity;
- frozen chronological boundaries;
- leakage-safe windowing;
- train-only preprocessing enforcement;
- tested configurable SAE implementation;
- reproducible experiment configuration/run specification.
