# Detector Selection Experiment Protocol

## 1. Experiment Objective
Define exactly how candidate anomaly detectors will be compared and selected using strictly the TRAIN and CALIBRATION partitions. The goal is to select the most performant detector configuration—not an alert threshold or operational policy—without leaking information from the protected FINAL HOLDOUT.

## 2. Candidate Space
The allowed search dimensions strictly consist of 16 combinations:
* **WINDOW (W)**: {6, 30, 90, 180}
* **SCALER**: {StandardScaler, MinMaxScaler}
* **SPARSITY**: {L1, KL}

No other hyperparameters, architectures, or search dimensions are permitted. 

## 3. Controlled Variables (Fairness Contract)
To ensure a mathematically fair comparison, all candidate experiments will share identical invariant conditions:
- **Sensors**: Same 7 input sensors
- **Data Partitions**: Same frozen chronological boundaries
- **Window Rules**: Identical causal-window logic and service-gap exclusions (Δt > 60s)
- **Stride**: 1
- **Seed**: 42 (unless executing the final reproducibility check)
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Batch Size**: 256
- **Epoch Budget**: 10
- **Train-Validation Method**: 10% chronological derived from TRAIN
- **Labeling Convention**: End-timestamp aligned labels
- **Metrics**: Standardized evaluation metrics

## 4. Data Partition Rules
- **TRAIN**: Used exclusively for model weight updates and scaler fitting.
- **TRAIN-derived Validation**: Used exclusively for early stopping and restoring the best training state.
- **CALIBRATION**: Used exclusively for candidate performance evaluation and final model selection.
- **FINAL HOLDOUT**: Completely inaccessible. No calibration data is used for weights; no holdout data is used for selection.

## 5. Metrics
**Primary Ranking Metrics:**
1. Average Precision (AP)
2. ROC-AUC

**Secondary Diagnostics:**
- Trapezoidal PR-AUC
- Normal score median, p95, p99
- Failure score median, p50, p75, p95
- Event-level score statistics
- Training/validation loss stability
- Runtime
- Usable-window and exclusion counts

## 6. Event-Level Evaluation
Because calibration contains exactly two distinct failure modes (Event_1 and Event_2), aggregate failure metrics can be misleading if a model only detects one event perfectly. Each candidate must report failure-window counts, mean, median, p25, p75, p90, p95, and p99 independently for Event_1 and Event_2. Success is contingent on detecting both.

## 7. Configuration-Selection Rule
Configuration selection operates via a deterministic, pre-defined procedure:
1. **Primary Criterion**: Maximum Average Precision on the Calibration set. Average Precision properly penalizes models that fail to handle the extreme 2.69% failure prevalence.
2. **Quality Criterion**: Both Event_1 and Event_2 median scores must be strictly higher than the normal p99 score. The training curves must demonstrate stable convergence (no diverging validation loss).
3. **Tie-Break**: If Average Precision is statistically indistinguishable, select the configuration with lower model complexity (smaller W, L1 over KL) and better worst-event performance.

## 8. Proposed Experiment Order
To efficiently navigate the 16 combinations, the following orthogonal sequence is proposed:
1. **Scaler Selection**: Fix W=30, Sparsity=L1. Run MinMaxScaler. Compare against the StandardScaler baseline. (Justification: Normalization fundamentally affects continuous input distributions and network convergence).
2. **Sparsity Selection**: Fix W=30, use the winning Scaler. Run KL. Compare against L1. (Justification: Latent regularization changes representational capability without altering temporal capacity).
3. **Temporal Selection**: Fix winning Scaler and Sparsity. Sweep W ∈ {6, 90, 180}. Compare against W=30. (Justification: Varying W affects data availability via exclusions; standardizing convergence rules first isolates the temporal effect).

## 9. Reproducibility Policy
Initial screening of candidates relies strictly on Seed 42. Computing 16 configurations across multiple seeds initially multiplies compute wastefully. Once a winning configuration is identified by the selection rule, a small repeatability check (e.g., 5 random seeds) will be executed strictly on the shortlisted model to ensure the Average Precision improvement over the baseline is statistically stable.

## 10. Resource Estimate
- **Candidate Runs**: Maximum 16, realistically 5-7 following the optimized order.
- **Windows per W**: Approximately ~400k for W=30, dropping slightly for W=180 due to initial W-1 exclusions.
- **Workload**: ~400,000 windows * 10 epochs.
- **CPU Time**: ~200 seconds per candidate (extrapolated from baseline v2).
- **Total Compute**: < 1 computing hour for all 16 runs.
- **RAM**: < 4 GB peak.

## 11. Artifact Requirements
Every candidate run must strictly save an artifact containing:
- run_id, config hash, dataset fingerprint, git commit, seed, configuration values
- partition counts, window accounting
- full 10-epoch training history
- Average Precision, ROC-AUC, trapezoidal PR-AUC
- score distributions (Normal, Failure)
- event-level statistics (Event_1, Event_2)
- runtime, environment versions

## 12. No Threshold Yet
No candidate experiment during this phase is permitted to select an anomaly threshold, optimize persistence rules, or build an alert layer. These operational steps are exclusively reserved for post-selection tuning.

## 13. Stop Conditions (Final Holdout)
FINAL HOLDOUT remains completely sealed. It cannot be loaded, evaluated, or used for tie-breaking.
