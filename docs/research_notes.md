# MetroGuard — Research Notes & Methodology Audit

## Summary

This document presents a rigorous methodology audit of the primary research literature underpinning MetroGuard, audits the evidence base, investigates secondary public implementations, and delineates confirmed facts from unstated reproduction details.

## 1. Original Research (Davari et al., DSAA 2021)

**Primary Reference:**  
Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021). *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.* 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), pp. 1–10. DOI: `10.1109/DSAA53316.2021.9564181`.

### Explicit Primary-Source Classifications

Every technical aspect of the baseline is audited and classified below into one of three strict evidentiary categories:

| Methodology Item | Evidentiary Status | Primary Source Evidence & Details |
|---|---|---|
| **Input Sensors** | `PRIMARY PAPER — explicitly stated` | Evaluates APU sensor data containing pressures, temperature, motor current, and valve control signals. |
| **Analog vs Digital Inputs** | `PRIMARY PAPER — explicitly stated` | Evaluates two distinct SAE models: one on analogue sensors (found most effective for air leak detection) and one on digital sensors. |
| **Preprocessing & Cleaning** | `NOT SPECIFIED IN PRIMARY PAPER` | Specific outlier removal, gap handling, or missing data treatments are not specified. |
| **Feature Construction** | `NOT SPECIFIED IN PRIMARY PAPER` | No cycle-derived features (e.g. duty cycles, cycle duration) or manual feature engineering are described; inputs are raw sensor channels. |
| **Normalization / Scaling** | `PRIMARY PAPER — reasonably inferable` | Inputs to autoencoders are scaled to bounded ranges for stable reconstruction, but exact scaling method (Min-Max bounds or Z-score) is unstated. |
| **Temporal Representation** | `NOT SPECIFIED IN PRIMARY PAPER` | Discusses sequence modeling of time-series telemetry, but does not state window length $W$, stride $S$, or tensor format. |
| **Sequence / Window Construction** | `NOT SPECIFIED IN PRIMARY PAPER` | Exact window parameters (e.g. 30 vs 60 timesteps) are not specified. |
| **SAE Architecture** | `PRIMARY PAPER — reasonably inferable` | Symmetric encoder-decoder with dense bottleneck layer, but layer count and hidden units are not explicitly tabulated. |
| **Latent Representation** | `PRIMARY PAPER — explicitly stated` | Bottleneck latent space trained to capture the normal operational dynamics of the APU. |
| **Sparsity Mechanism** | `NOT SPECIFIED IN PRIMARY PAPER` | Sparse autoencoder is specified, but the exact sparsity regularizer (L1 activity penalty vs KL divergence) and penalty weight $\lambda$ are unstated. |
| **Training Data Partition** | `PRIMARY PAPER — explicitly stated` | Trained exclusively on normal operating data without failure events (exact calendar dates not specified). |
| **Anomaly Score** | `PRIMARY PAPER — explicitly stated` | Mean Squared Error (MSE) reconstruction error between input and autoencoder output. |
| **Thresholding** | `PRIMARY PAPER — reasonably inferable` | Compared against a threshold derived from normal operating reconstruction error; exact quantile or formula unstated. |
| **Filtering / Post-Processing** | `NOT SPECIFIED IN PRIMARY PAPER` | No moving average, temporal persistence ($K$-consecutive), or cooldown filter is defined; pointwise errors are evaluated directly. |
| **Alarm Semantics** | `NOT SPECIFIED IN PRIMARY PAPER` | Transition from pointwise error spikes to an operational maintenance alert episode is not defined. |
| **Evaluation Protocol** | `PRIMARY PAPER — explicitly stated` | Overlap between predicted failure intervals and maintenance-reported failure intervals (TP, FP, TN, FN). Target: detection $\ge 2$ hours prior to functional failure. |
| **Reported Metrics** | `PRIMARY PAPER — explicitly stated` | Accuracy, Precision, Recall, F-measure, and detection lead time. |

## 2. Dataset Evidence

- **Authoritative Source:** UCI Machine Learning Repository (Dataset #791, MetroPT-3).
- **Verified Facts:** 1,516,948 instances, 15 sensor attributes (7 analogue + 8 digital), zero null values, zero duplicate timestamps.
- **Cadence Reconciliation:** While local companion documentation states 1 Hz (and displays 15,169,480 instances due to a typographical extra zero), actual local data has 1,516,948 rows with 88.17% of timestamp differences at 10.0 seconds (1,337,521 intervals), 8.46% at 9.0s (128,277 intervals), and 331 intervals $>60$ seconds. The effective sampling rate is ~0.1 Hz (10s intervals).
- **Service Gaps:** 331 intervals $>60$ seconds (maximum gap: 48.03 hours).
- **Ground Truth:** 4 air leak intervals in April, May, June, and July 2020 totaling 29,960 rows (1.975% of dataset).

## 3. Secondary Reproductions & The "TP3 + Motor Current" Audit

### Audit of the "TP3 + Motor Current" Claim
- **Audit Question:** Is the Davari baseline defined as an autoencoder trained only on `TP3` (pneumatic panel pressure) and `Motor Current`?
- **Finding:** **NO.** The primary paper (Davari et al., 2021) contains no such limitation. Davari et al. evaluated the analogue sensor group (all 7 analogue signals) for air leaks, not a 2-variable subset.
- **Source of the Claim:** The combination of `TP3` and `Motor Current` appears in secondary student theses, academic case studies (e.g., Charlotte / UVigo / UPV), and Medium tutorials as an illustrative bivariate example because pressure drop and current draw physically illustrate compressor loading.
- **Classification:** SECONDARY exploratory evidence. It is a downstream modeling choice, NOT the original Davari research baseline.
- **Rule:** MetroGuard must NOT claim or imply that `Davari baseline = TP3 + Motor Current`.

### Audit of Cycle-Derived Features
- Claims in some predictive maintenance literature regarding "cycle segmentation," "duty cycle percentage," or "fill-time trends" are secondary feature engineering techniques. They are NOT part of the Davari et al. (2021) sparse autoencoder formulation.

### Community Implementation Flaws
Public repositories (e.g., `chirag-9121/predictive-maintenance` and Kaggle notebooks) reveal common methodological flaws:
- Applying random `train_test_split` to time-series data, introducing extreme future-to-past data leakage.
- Disregarding multi-hour operational service breaks during windowing.
- Reporting point-wise classification accuracy (>98%) on imbalanced data without measuring false alarms per day or detection lead time.

## 4. Evidence Matrix Across Sources

| Topic | Original Davari Paper | Dataset / UCI | Secondary Sources | MetroGuard Stance |
|---|---|---|---|---|
| **Sensors** | Analogue vs digital evaluated in separate SAEs (Explicit) | 15 sensor features (7 analogue, 8 digital) (Explicit) | Often selects 2 features (TP3 + Current) or all 15 blindly | Primary baseline evaluates 7 analogue sensors; controlled ablation evaluates all 15 |
| **Preprocessing** | Normalized/scaled (Inferred); no gap rules (Unresolved) | Raw sensor readings; no nulls (Explicit) | MinMax / StandardScaler; often random split (Leakage) | Causal scaling fitted strictly on training partition; zero leakage; drop `column00` |
| **Windowing** | Sequence modeling mentioned; $W$ and $S$ unstated (Unresolved) | Nominal 10s cadence with 331 physical gaps > 60s (Explicit) | Pointwise (1 row) to arbitrary windows (30/60 rows) without gap checks | Causal past-only windowing with mandatory gap-reset rule when $\Delta t > 60\text{s}$; window size tested as controlled parameter |
| **SAE Architecture** | Symmetric encoder-decoder with latent bottleneck (Inferred) | N/A (Data only) | Various dense or LSTM configurations | Documented, reproducible symmetric MLP autoencoder with fixed random seeds |
| **Sparsity** | Sparse autoencoder stated; exact penalty unstated (Unresolved) | N/A | L1 activity regularization on bottleneck | Explicitly parameterize L1 activity penalty ($\lambda \sum |z|$) on latent activations |
| **Anomaly Score** | MSE reconstruction error (Explicit) | N/A | MSE or MAE | Squared reconstruction error $\|x - \hat{x}\|_2^2$ per window |
| **Threshold** | Normal distribution threshold; exact value unstated (Inferred) | N/A | Arbitrary quantiles (95th, 99th) or heuristics | Calibrated strictly on calibration set (percentile or cost objective); never on holdout |
| **Filtering** | None specified; pointwise scores used (Unresolved) | N/A | Rolling mean or none | Explicit temporal smoothing (moving average or persistence window $K$) |
| **Alarm Semantics** | Interval overlap evaluated; state machine unstated (Unresolved) | 4 failure intervals in maintenance reports (Explicit) | Pointwise flags without episode logic | Operational state machine (`NORMAL`, `ALERT`, `DATA_QUALITY_ISSUE`) with persistence and cooldown |
| **Evaluation** | Interval overlap, Precision, Recall, Lead time $\ge 2\text{h}$ (Explicit) | Interval start/end ground truth (Explicit) | Often pointwise accuracy / ROC-AUC (Misleading) | Event-level recall, detection lead time, false alarms per day, alert occupancy, protected final holdout |

## 5. MetroGuard Interpretation

1. The baseline is constrained to confirmed technical facts (unsupervised sparse autoencoder on healthy normal data, reconstruction error anomaly score, interval-based evaluation).
2. Unspecified reproduction choices (layer dimensions, window length $W$, sparsity weight $\lambda$, threshold quantile) must be treated as controlled calibration parameters, not assumed ground truth.
3. The core contribution of MetroGuard is explicitly located in the **operational decision layer**:
   - Cost-aware alert calibration.
   - Operating-regime-aware policy.
   - Input-quality gating.
   - Leakage-safe chronological evaluation framework.

## 6. Open Questions for ChatGPT Review

1. **Baseline Feature Set:** Should the initial baseline reproduction use the 7 analogue sensors (aligned with Davari's air leak findings) or all 15 sensors?
2. **Sparsity Regularization:** Confirm adoption of L1 bottleneck activity regularization as the primary sparsity mechanism.
3. **Chronological Partition Dates:** Review candidate split (Train: Feb–Mar normal; Calibration: Apr–May with Events 1 & 2; Protected Holdout: Jun–Aug with Events 3 & 4).
4. **Causal Window Size:** Select initial window horizon $W$ (e.g. 30 samples = 5 minutes at 10s cadence) for calibration.
