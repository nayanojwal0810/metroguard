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

The final window size is not frozen yet. A candidate calibration set is defined at the nominal ~10-second empirical cadence:

- **6 observations** $\approx$ 1 minute
- **30 observations** $\approx$ 5 minutes
- **90 observations** $\approx$ 15 minutes
- **180 observations** $\approx$ 30 minutes

These window sizes represent candidate engineering configurations, not claims from literature. The final window size will be selected based strictly on permitted training and calibration evidence; the final holdout must not influence window selection.

### Provisional Causal Window Rule

To ensure leakage-safe operational behavior:
1. **Strictly Causal:** For any decision at time $t$, window features use only observations available at or before $t$. Centered rolling windows and future backfilling are strictly forbidden.
2. **Gap Boundary Rule:** Windows cannot cross operational service or telemetry breaks. A gap-reset condition is established at $\Delta t > 60\text{ seconds}$.
3. **Engineering Provenance:** The 60-second gap boundary is an empirical MetroGuard engineering rule, not a literature fact.
4. **Insufficient History:** Periods following a service break with fewer observations than window length $W$ must be handled explicitly (e.g. state flagged as warming up or insufficient history) rather than padded using future observations or interpolated across gaps.

## Preprocessing Requirements

Only preprocessing requirements supported by verified data characteristics are mandated:
- **Index Removal:** Drop the serialized row index column `column00`.
- **Causal Scaling:** Normalization (e.g. Robust or MinMax scaling) parameters must be fitted strictly on the training partition and applied causally to downstream data.
- **Gap Isolation:** Timestamp differences $\Delta t$ must be evaluated sequentially to reset window buffers across service breaks ($>60\text{s}$).

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
