# MetroGuard — Baseline Specification

## Purpose

This document provides the authoritative technical baseline specification for MetroGuard. Following a rigorous methodology audit, it strictly categorizes baseline attributes into confirmed facts from the primary paper, secondary implementation evidence from external sources, and unresolved reproduction decisions requiring explicit technical review.

## 1. Confirmed Baseline Facts

These elements are directly supported by the primary research paper (Davari et al., DSAA 2021):

- **Model Paradigm:** Unsupervised deep anomaly detection using a **Sparse Autoencoder (SAE)**.
- **Architectural Principle:** Symmetrical encoder-decoder with a lower-dimensional latent bottleneck representing the normal operational state of the compressor.
- **Training Strategy:** The network is fitted exclusively on healthy normal operational telemetry without known failures or maintenance actions.
- **Objective Function:** Reconstruction error loss combined with a sparsity constraint on the latent hidden layer:
  $$\mathcal{L}(X, \hat{X}) = \text{MSE}(X, \hat{X}) + \Omega(Z)$$
- **Pointwise Anomaly Score:** Instantaneous reconstruction error computed via Mean Squared Error (MSE):
  $$s_t = \frac{1}{D} \sum_{i=1}^D (x_{t,i} - \hat{x}_{t,i})^2$$
- **Thresholding Principle:** Anomaly flags are triggered when reconstruction error exceeds a threshold derived from normal operational data.
- **Evaluation Criteria:** Detection success is evaluated against maintenance-reported failure intervals (aiming for early warning $\ge 2$ hours before functional failure) using event-level overlap (TP, FP, TN, FN).

## 2. Secondary Implementation Evidence

The following elements originate from secondary literature, community implementations (e.g. `chirag-9121/predictive-maintenance`), academic case studies, or exploratory analyses. They are NOT confirmed parts of the primary Davari paper and must not be conflated with the original baseline:

- **The "TP3 + Motor Current" Subsetting:**
  - *Source:* Secondary academic theses and online tutorials (e.g. Charlotte, UVigo, Medium).
  - *Status:* Exploratory bivariate simplification chosen for visual clarity (pressure drop vs motor current draw). It is NOT the Davari baseline.
- **Cycle-Derived Feature Engineering:**
  - *Source:* General predictive maintenance literature on reciprocating compressors.
  - *Status:* Secondary feature engineering not utilized in Davari et al. (2021).
- **Sparsity Implementation (L1 Regularization):**
  - *Source:* Standard Keras / PyTorch Sparse Autoencoder conventions.
  - *Status:* Commonly implemented as an L1 activity regularization penalty on the bottleneck activations ($\lambda \sum |z|$), though KL divergence is also used in deep learning literature.
- **Random Train/Test Splitting:**
  - *Source:* Found in several open-source community notebooks.
  - *Status:* Scientifically invalid for time-series data; introduces severe temporal data leakage and is strictly rejected by MetroGuard.

## 3. Reproduction & Baseline Design Decisions

These technical choices are tracked for MetroGuard baseline implementation:

- **Input Feature Subset:**
  - *Status:* Approved via METROGUARD ENGINEERING DECISION (`DEC-008`). The primary baseline uses the 7 continuous analogue sensors (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`). The 15-sensor set is reserved for future controlled ablation.
- **Temporal Window Representation ($W$ and $S$):**
  - *Status:* Candidate calibration set defined: $\{6, 30, 90, 180\}$ observations ($\approx 1\text{m}, 5\text{m}, 15\text{m}, 30\text{m}$ at nominal ~10s cadence). Final window selection will be made on calibration evidence; final holdout remains protected.
- **Service Gap Reset Boundary:**
  - *Status:* Approved provisional causal window rule: windows use only observations available at or before $t$, and cannot cross telemetry breaks. A gap-reset condition is established at $\Delta t > 60\text{s}$ (MetroGuard engineering rule).
- **Sparsity Penalty Formulation & Weight ($\lambda$):**
  - *Status:* Unresolved. Choice between L1 bottleneck activity regularization versus KL divergence penalty, along with the numerical weight $\lambda$.
- **Network Depth and Hidden Dimensions:**
  - *Status:* Unresolved. Exact layer counts and hidden unit dimensions (e.g., $D \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 16 \rightarrow 32 \rightarrow D$) must be documented as an engineering choice.
- **Threshold Selection Criterion:**
  - *Status:* Unresolved. Exact statistical quantile (e.g., 99th or 99.5th percentile of normal calibration loss) or cost-optimal threshold selection.
- **Alert Persistence & Smoothing ($K$, Cooldown):**
  - *Status:* Unresolved. Establishing minimum consecutive violations ($K$) and cooldown duration to convert pointwise reconstruction errors into stable operational maintenance alert episodes.
- **Chronological Split Dates:**
  - *Status:* Unresolved. Candidate boundaries (Train: Feb–Mar; Calibration: Apr–May; Final Holdout: Jun–Aug) require formal confirmation.
