# MetroGuard — Current State

## Active objective

Research and dataset verification before ML implementation.

## Current status

Repository workspace structure is established:
- Git is initialized on `main`.
- `origin` points to `https://github.com/nayanojwal0810/metroguard.git`.
- Control documents are established at the repository root.
- Raw dataset `data/raw/MetroPT3(AirCompressor).csv` is present and protected from Git via `.gitignore`.
- Implementation has not started.

## Approved project direction

- MetroPT-3 is the primary dataset.
- Sparse autoencoder is the primary baseline formulation.
- MetroGuard focuses on anomaly-to-maintenance decisioning.
- Cost-aware alerting, regime-aware policy, and input-quality gating are the approved modifications.
- Chronological leakage-safe evaluation is mandatory.
- Final holdout remains protected from tuning.
- CPU and Colab share one Git-controlled codebase.

## Not yet frozen

- dataset hash/version;
- exact feature set;
- causal window/aggregation design;
- chronological boundaries;
- baseline architecture details;
- baseline threshold method;
- alert episode semantics;
- cost function/weights;
- regime definition;
- quality thresholds;
- monitoring triggers;
- serving/tracking technologies.

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
