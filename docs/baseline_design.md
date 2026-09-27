# MetroGuard — Baseline Design Specification

## Objective

The primary baseline is an unsupervised anomaly detection system designed to detect developing compressor air leaks and abnormal pneumatic behavior from continuous sensor telemetry before functional system failure occurs.

## Input

**METROGUARD ENGINEERING DECISION:**  
The approved primary baseline input consists of the seven continuous analogue sensors present in the local MetroPT-3 dataset:

1. `TP2` — Pressure on compressor (bar)
2. `TP3` — Pneumatic panel pressure (bar)
3. `H1` — Cyclonic separator filter pressure drop (bar)
4. `DV_pressure` — Air dryer discharge pressure drop (bar)
5. `Reservoirs` — Downstream reservoir tank pressure (bar)
6. `Oil_temperature` — Compressor oil temperature (°C)
7. `Motor_current` — Compressor motor electric current (A)

*Note:* This input set is a MetroGuard engineering decision based on domain relevance to air leaks. It is not an exact literature specification, nor is it a bivariate `TP3 + Motor Current` model. The 15-sensor representation (incorporating the 8 digital signals) is reserved for a future controlled feature ablation.

## Temporal Representation

The final window size is not frozen yet. A candidate calibration set is defined based on observation counts:

- **6 observations** (nominal scale: ~1 minute at ~10s cadence)
- **30 observations** (nominal scale: ~5 minutes at ~10s cadence)
- **90 observations** (nominal scale: ~15 minutes at ~10s cadence)
- **180 observations** (nominal scale: ~30 minutes at ~10s cadence)

**Authoritative Window Definition:**
- A window contains exactly $W$ consecutive observations.
- The window ends at the evaluation observation $t_i$.
- All observations are at or before $t_i$ ($t_{i - W + 1} \le \dots \le t_i$).
- Every consecutive timestamp gap within the window must be $\le 60\text{ seconds}$.
- A window may not cross a train/calibration/holdout boundary.
- A window may not cross a service gap.
- No padding, interpolation, or synthetic observations are introduced.
- **Therefore, $W$ represents observation count, not an exact elapsed-time duration.** Physical duration varies naturally with sampling cadence jitter (8–13s).

### Window Construction Rules

To ensure leakage-safe operational behavior:
1. **Strictly Causal:** For any decision at time $t_i$, window features use only observations available at or before $t_i$: $[t_{i - W + 1}, \, t_i]$. Centered rolling windows and future backfilling are strictly forbidden.
2. **Fixed Observation Count:** Each window contains exactly $W$ continuous consecutive observations.
3. **Configurable Stride:** Stride $S$ is configurable (default $S=1$ for continuous inference, $S \ge 1$ for downsampled evaluation).
4. **Gap Boundary Rule:** Windows cannot cross operational service or telemetry breaks. A gap-reset condition is established at $\Delta t > 60\text{ seconds}$.
5. **Split Boundary Rule:** Windows cannot cross chronological split boundaries ($\text{Train} \to \text{Calibration} \to \text{Holdout}$).
6. **Insufficient History:** Periods following a service break or partition boundary with fewer observations than window length $W$ must produce no model-ready window (`INSUFFICIENT_HISTORY`), rather than using fabricated padding or bridging across gaps.

## Preprocessing Requirements

The preprocessing pipeline strictly separates parameter fitting from evaluation transforms:

```text
raw CSV
 ↓
timestamp validation & ordering
 ↓
select primary features (7 continuous analogue channels)
 ↓
chronological split (Train / Calibration / Final Holdout / Unused Tail)
 ↓
fit preprocessing ONLY on training data
 ↓
transform calibration / holdout using training-fitted parameters
 ↓
causal temporal windows (with gap and split boundary isolation)
 ↓
model-ready representation
```

### Preprocessing Invariants

- **Index Removal:** Drop the serialized row index column `column00`.
- **Fit Isolation:** Preprocessing parameters (mean, standard deviation, minimum, maximum) are fitted exclusively on the `TRAIN` partition (`2020-02-01 00:00:00` to `2020-03-31 23:59:59`).
- **Zero Leakage:** Under no circumstances is a scaler fitted separately on calibration or holdout data, nor on the full combined dataset.
- **No Future Imputation:** Future observations must never be used to interpolate or fill missing history.
- **Gap Isolation:** Timestamp differences $\Delta t$ must be evaluated sequentially to reset window buffers across service breaks ($>60\text{s}$).

### Candidate Scaling Methods

The choice of scaling method is not frozen intuition; candidate methods will be compared using training/calibration evidence (final holdout remains protected):

1. **Standard Scaling ($Z$-Score):**
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   Zero-centered, unit-variance representation. Sensitive to extreme outliers; preserves relative pneumatic scale.
2. **Min-Max Scaling:**
   $$\tilde{x} = \frac{x - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}$$
   Bounds healthy training data into $[0, 1]$. Values during failure states may exceed $[0, 1]$, signaling departure from normal operating range.

*Note:* Arbitrary clipping bounds are not implemented at this stage; out-of-range behavior will be assessed objectively during calibration.

## Model Requirements

The baseline model is a **Sparse Autoencoder (SAE)**:
- **Architecture:** Symmetrical encoder-decoder structure with a lower-dimensional latent bottleneck representing the healthy operating manifold of the compressor.
- **Sparsity Constraint:** An explicit sparsity regularizer on the latent activations to constrain model capacity and prevent trivial identity mapping.
- **Training Paradigm:** Fitted exclusively on verified normal operational telemetry without known failures or maintenance interventions.

Undocumented layer counts, neuron numbers, and activation parameters from external literature are not assumed; exact architecture dimensions remain open for calibration.

## Anomaly Score

The pointwise anomaly score is the Mean Squared Error (MSE) reconstruction loss across the input dimension $D$:

$$s_t = \frac{1}{D} \sum_{i=1}^D (x_{t,i} - \hat{x}_{t,i})^2$$

## Thresholding

The anomaly threshold $\tau$ will be calibrated strictly from permitted normal/calibration data. No threshold value or quantile is frozen yet. The final holdout partition will not be accessed for threshold tuning.

## Evaluation

Evaluation adheres to the approved leakage-safe chronological structure:
$$\text{Training} \longrightarrow \text{Calibration} \longrightarrow \text{Protected Final Holdout}$$

- **Operational Metrics:**
  - Detection lead time prior to functional failure (target: $\ge 2$ hours).
  - Event-level early detection recall across ground-truth failure intervals.
  - False alarm rate (false alarms per day).
  - Operational alert occupancy (percentage of time spent in alert state).

## Open Decisions

The following technical decisions remain open and will be resolved through documented evidence:
- **exact window**: Selection among candidate set {6, 30, 90, 180} observations based on calibration evidence;
- **scaling method**: Choice of normalization/scaling method and clipping boundaries;
- **SAE architecture**: Network depth, layer dimensions, and bottleneck size;
- **sparsity mechanism**: Sparsity regularizer formulation (L1 activity penalty vs KL divergence) and penalty weight $\lambda$;
- **training configuration**: Optimizer, learning rate, batch size, epochs, and early stopping rules;
- **threshold**: Selection method and value derived exclusively from normal calibration loss;
- **alert semantics**: Persistence requirement ($K$), smoothing mechanism, and cooldown duration.
