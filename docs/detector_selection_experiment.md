# Detector Selection Experiment Protocol

## 1. Why the Full 16-Run Matrix is Being Used
The initial assumption that scaler, sparsity, and window effects can be optimized sequentially without important interaction effects lacks empirical evidence. The predefined candidate space consists of only 4 (W) × 2 (Scaler) × 2 (Sparsity) = 16 configurations. Executing the complete 16-configuration factorial matrix provides stronger evidence and eliminates the risk of missing critical interaction effects.

## 2. Controlled Variables
Every candidate run will use exactly the same:
- 7 analogue input features
- Chronological TRAIN / CALIBRATION partitions
- Train-fit / train-validation methodology (0.10 validation fraction)
- Causal windowing
- Δt > 60 second service-gap rule
- Stride = 1
- Adam optimizer, learning rate = 1e-3, batch size = 256, epoch budget = 10
- Seed = 42
- End-timestamp labeling convention
- Artifact schema and invariant model capacity dimension rule (deterministic H/Z per W)
The only variables allowed to change are W, Scaler, and Sparsity. The matrix evaluates predefined L1 and KL regularization configurations using their existing baseline documented lambda values.

## 3. Primary Metric
The primary detector-ranking metric is **Average Precision** (`sklearn.metrics.average_precision_score`), which accurately evaluates precision-recall trade-offs under severe class imbalance without relying on arbitrary operational thresholds.

## 4. Secondary Metrics
Secondary metrics include:
- `pr_auc_trapezoidal`
- `roc_auc`
- best validation epoch, best validation loss, final validation loss, final train loss
- parameter count and runtime
- normal median, failure median, normal p95/p99, failure p50/p75/p95

## 5. Event-Level Evaluation
Event-level performance is analyzed descriptively for Event_1 and Event_2 to ensure aggregate metrics do not hide single-event blindness. We report:
- usable failure-window count, mean, median, p25, p75, p90, p95, p99
- Event_1 one-vs-rest Average Precision (descriptive event-level diagnostic)
- Event_2 one-vs-rest Average Precision (descriptive event-level diagnostic)

Event-specific AP is explicitly a descriptive diagnostic and must not replace global Average Precision as the primary selection metric, nor create a separate event-based selection objective.

## 6. Validity Rules
A configuration is strictly invalid if it exhibits:
- failed training (e.g., NaN/Inf loss)
- non-finite losses
- an invalid artifact
- missing required calibration metrics
- insufficient usable calibration windows for the protocol

## 7. Exact Selection Rule
**Stage A (Validity)**: Filter out all invalid configurations.
**Stage B (Primary Ranking)**: Rank valid configurations descending by Average Precision on CALIBRATION.
**Stage C (Near-Tie Handling)**: See section 8.

## 8. Near-Tie Definition
Define a near tie for a candidate as:
`top_average_precision - candidate_average_precision <= 0.005`

This is a practical tolerance, not a statistical significance test. Candidates inside the top near-tie band are then compared using:
1. Highest worst-event Average Precision
2. Training stability
3. Lower parameter count

## 9. Repeatability Protocol
After identifying the top candidate and the runner-up (if within the AP <= 0.005 band) from the seed-42 screen, a reproducibility check is run strictly on those shortlisted candidates using seeds {7, 21, 42}. The check reports mean AP, AP std dev, mean ROC-AUC, worst-event AP across seeds, and runtime. No holdout data is accessed.

## 10. Freeze Procedure
Following the repeatability check, exactly one detector configuration is selected according to the pre-declared evaluation protocol. Recorded metadata will include W, scaler, sparsity, λ, architecture dimensions, parameter count, training configuration, seed policy, dataset fingerprint, and Git commit. No further tuning is allowed. FINAL HOLDOUT remains untouched.

## 11. PRE-RUN ESTIMATES: Resource & Batch Accounting by W
Exact window and batch counts depend strictly on W, stride, service-gap locations, and partition boundaries. Each artifact will record exact counts during execution. For planning purposes, approximate pre-run estimates are:

**Batch Accounting Example (Baseline W=30):**
- TRAIN-fit windows: ~398,594
- Batch size: 256
- Batches per epoch: ceil(398,594 / 256) = 1,558
- Total training batches across all epochs (10): 15,580
- Validation batches: ceil(44,384 / 256) = 174
- Calibration inference batches: ceil(408,431 / 256) = 1,596

**Resource Estimates by W:**
Based on the current implementation, all unscaled TRAIN-fit, TRAIN-val, and CALIBRATION windows are retained in memory simultaneously. During scaling, float64 intermediate arrays and PyTorch float32 copies are additionally created.
- **W=6**: Peak window storage ~141MB. Total peak RAM < 1GB. CPU Runtime: ~180s per run.
- **W=30**: Peak window storage ~710MB. Total peak RAM ~2GB. CPU Runtime: ~200s per run.
- **W=90**: Peak window storage ~2.1GB. Total peak RAM ~5-6GB. CPU Runtime: ~230s per run.
- **W=180**: Peak window storage ~4.2GB. Scaler intermediate and PyTorch dataloader copies increase this drastically. Total peak RAM ~10-12GB. CPU Runtime: ~280s per run.

## 12. PRE-RUN ESTIMATES: Total Estimated 16-Run Cost
The arithmetic for the complete factorial matrix execution cost is:
- W=6: ~180s * 4 combinations = 720s
- W=30: ~200s * 4 combinations = 800s
- W=90: ~230s * 4 combinations = 920s
- W=180: ~280s * 4 combinations = 1120s

**Total CPU Runtime:** 720 + 800 + 920 + 1120 = 3,560 seconds (~1 CPU hour).
All estimated runtimes include validation and calibration inference. Observed runtime, peak RAM, and batch counts will be recorded in artifacts for subsequent operational planning.

## 13. Matrix Summary Artifact Design
A lightweight matrix summary artifact will be saved containing one row per candidate: `run_id`, `w`, `scaler`, `sparsity`, `sparsity_weight`, `h`, `z`, `parameter_count`, `average_precision`, `pr_auc_trapezoidal`, `roc_auc`, `event_1_ap`, `event_2_ap`, `event_1_median`, `event_2_median`, `normal_median`, `normal_p95`, `normal_p99`, `best_epoch`, `best_val_loss`, `runtime`, `usable_calibration_windows`, `exclusion_counts`.

## 14. No Threshold Yet
No candidate experiment during this phase is permitted to select an anomaly threshold, optimize persistence rules, or build an alert layer. These operational steps are exclusively reserved for post-selection tuning.

## 15. Stop Conditions (Final Holdout)
FINAL HOLDOUT remains completely sealed. It cannot be loaded, evaluated, or used for tie-breaking.
