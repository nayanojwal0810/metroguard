# MetroGuard

MetroGuard is an operational early-warning system designed to detect anomalous compressor behavior in metro train fleets and convert continuous anomaly scores into defensible maintenance-alert decisions. Operating on real MetroPT compressor telemetry, the system implements a leakage-safe time-series anomaly detection pipeline with an operational decision layer—incorporating cost-aware alerting, operating-regime awareness, and telemetry quality gating—supported by production-style monitoring, candidate retraining, and rollback lifecycle controls.

## Implementation Status

Repository setup and clean workspace preparation are complete. Implementation of the machine learning pipeline, feature engineering, and model training has not yet started.

## Technical Architecture

```text
Metro Telemetry
       ↓
Input Data-Quality Gate
       ↓
Causal Preprocessing & Windowing
       ↓
Leakage-Safe Anomaly Detection (Sparse Autoencoder)
       ↓
Anomaly Score
       ↓
Operational Decision Layer (Regime Determination & Cost-Aware Alert Policy)
       ↓
Operational State (NORMAL / ALERT / DATA_QUALITY_ISSUE)
       ↓
Monitoring → Retraining Candidate Generation → Rollback Lifecycle
```

## Documentation & Project Control

For complete technical specifications, evaluation methodology, and project governance, refer to [execution_plan.md](execution_plan.md) and the project control documents in the repository root:

- [execution_plan.md](execution_plan.md) — Technical scope, experimental design, and completion criteria
- [rules.md](rules.md) — Operating rules, role boundaries, and stop conditions
- [decision_log.md](decision_log.md) — Approved technical and architectural decisions
- [current_state.md](current_state.md) — Active phase, work boundaries, and decision gates
- [python_design.md](python_design.md) — Code style, structure, and engineering standards
- [git_workflow.md](git_workflow.md) — Git integration between local and remote execution environments
- [colab.md](colab.md) — Execution rules for heavy workloads
- [document_design.md](document_design.md) — Documentation standards and reporting guidelines
- [docs/literature.md](docs/literature.md) — Concise reference table of literature reviewed during system design
