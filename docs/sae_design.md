# MetroGuard — Sparse Autoencoder (SAE) Specification

## 1. Purpose & Model Role

The primary baseline model for MetroGuard is an unsupervised **Sparse Autoencoder (SAE)**. Its purpose is to learn the healthy operational manifold of the compressor from normal continuous telemetry (`TRAIN` partition: February–March 2020) and flag anomalies through elevated reconstruction loss when pneumatic behavior deviates from normal operating dynamics (e.g. developing air leaks).

## 2. Input Representation

The model operates on flattened causal rolling windows constructed from the 7 primary continuous analogue sensors:
`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`.

For any candidate window size $W$, each window of shape $(W, 7)$ is flattened causally into a 1D feature vector:

$$x \in \mathbb{R}^D, \quad D = 7 \times W$$

Observations are ordered chronologically within the vector: the oldest observation in the window occupies indices $0 \dots 6$, and the current evaluation observation at time $t$ occupies indices $D-7 \dots D-1$.

## 3. Symmetric MLP Architecture

The baseline uses a compact, symmetric multilayer perceptron (MLP) autoencoder:

```text
Input Window: x ∈ R^D
       │
       ▼
Encoder Hidden: h_e = ReLU(W_e1 · x + b_e1) ∈ R^H
       │
       ▼
Latent Bottleneck: z = a_z(W_e2 · h_e + b_e2) ∈ R^Z
       │
       ▼
Decoder Hidden: h_d = ReLU(W_d1 · z + b_d1) ∈ R^H
       │
       ▼
Output Reconstruction: x̂ = W_d2 · h_d + b_d2 ∈ R^D
```

### Architectural Properties:
- **Hidden Layers:** Linear dense layers with Rectified Linear Unit (`ReLU`) activations in both encoder and decoder.
- **Output Layer:** Linear projection directly to $\mathbb{R}^D$ (no output activation; allows unconstrained reconstruction across standardized/scaled sensor domains).
- **Bottleneck Activation ($a_z$):**
  - Under **L1 Sparsity:** `ReLU` activation ($z \ge 0$), encouraging explicit zero-activation sparsity in latent representations.
  - Under **KL Divergence Sparsity:** `Sigmoid` activation ($z \in (0, 1)$), enabling interpretation of latent activations as Bernoulli firing probabilities.

## 4. Exact Dimensionality Rule

To provide a consistent, defensible baseline across all candidate window sizes without ad-hoc manual tuning, dimensions are assigned according to the following compact pyramidal compression rule:

$$H = \min\left(512, \, \max\left(32, \, 2^{\lfloor \log_2(D \times 0.6) \rfloor}\right)\right)$$

$$Z = \min\left(128, \, \max\left(16, \, 2^{\lfloor \log_2(H \times 0.35) \rfloor}\right)\right)$$

### Canonical Configuration Table:

| Window Size ($W$) | Input Dim ($D = 7 \times W$) | Hidden Dim ($H$) | Latent Dim ($Z$) | Bottleneck Compression ($D / Z$) | Total Parameter Count |
|---|---|---|---|---|---|
| **$W = 6$** | 42 | 32 | 16 | $2.63\times$ | 3,834 |
| **$W = 30$** | 210 | 128 | 32 | $6.56\times$ | 62,546 |
| **$W = 90$** | 630 | 256 | 64 | $9.84\times$ | 356,806 |
| **$W = 180$** | 1,260 | 512 | 128 | $9.84\times$ | 1,426,828 |

*Design Rationale:*
- Keeps parameter counts small enough for fast, deterministic CPU smoke tests and efficient GPU/Colab execution.
- Prevents overparameterization and trivial identity mappings through progressive bottleneck compression.
- The PyTorch module accepts explicit `input_dim`, `hidden_dim`, and `latent_dim` parameters; when dimensions are omitted, the canonical rule is applied automatically.

## 5. Loss Formulation & Candidate Sparsity Mechanisms

The total training loss consists of an explicit reconstruction MSE term and a regularized sparsity penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \Omega(Z)$$

### 1. Reconstruction Loss ($\mathcal{L}_{\text{recon}}$)

The reconstruction loss is the Mean Squared Error (MSE) between input $X$ and reconstruction $\hat{X}$:

$$\mathcal{L}_{\text{recon}}(X, \hat{X}) = \frac{1}{B \cdot D} \sum_{b=1}^B \sum_{j=1}^D (x_{b,j} - \hat{x}_{b,j})^2$$

This term is explicitly tracked and logged separately during training and evaluation.

### 2. Candidate Sparsity Mechanisms ($\Omega(Z)$)

Both candidate mechanisms are supported through the unified model interface:

#### Candidate A: L1 Latent Activity Regularization
Penalizes the mean absolute activation of the latent bottleneck:

$$\Omega_{\text{L1}}(Z) = \frac{1}{B \cdot Z} \sum_{b=1}^B \sum_{k=1}^Z |z_{b,k}|$$

- **Candidate Weight Range:** $\lambda \in [10^{-5}, 10^{-3}]$ (default: $10^{-4}$).
- **Characteristics:** Directly induces exact zeros in the ReLU bottleneck; simple, stable gradient flow.

#### Candidate B: Kullback-Leibler (KL) Divergence Penalty
Penalizes divergence between average latent activation $\hat{\rho}_k$ across the batch and a small target firing probability $\rho$:

$$\hat{\rho}_k = \frac{1}{B} \sum_{b=1}^B z_{b,k}, \quad z_{b,k} \in (0, 1) \text{ via Sigmoid}$$

$$\Omega_{\text{KL}}(\rho, \hat{\rho}) = \frac{1}{Z} \sum_{k=1}^Z \left[ \rho \log\left(\frac{\rho}{\hat{\rho}_k}\right) + (1 - \rho) \log\left(\frac{1 - \rho}{1 - \hat{\rho}_k}\right) \right]$$

- Numerical stability guard: $\hat{\rho}_k$ is clamped to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-6}$.
- **Candidate Weight Range:** $\lambda \in [10^{-3}, 10^{-1}]$ (default: $10^{-2}$).
- **Target Sparsity Parameter:** $\rho \in [0.01, 0.10]$ (default: $\rho = 0.05$).

*Status:* Neither sparsity mechanism is declared winner yet; selection will be performed using training/calibration evidence without accessing the final holdout.

## 6. Pointwise Anomaly Scoring

During inference, each evaluation window at time $t$ receives an instantaneous anomaly score equal to its sample-level reconstruction MSE:

$$s_t = \frac{1}{D} \sum_{j=1}^D (x_{t,j} - \hat{x}_{t,j})^2$$

This pointwise anomaly score feeds downstream alert policies (persistence $K$, moving smoothing, thresholding $\tau$).

## 7. Configuration Parameterization

The PyTorch implementation in `src/models/` is parameterized by:
- `input_dim`: Input feature dimension $D = 7 \times W$.
- `hidden_dim`: Encoder and decoder hidden layer width $H$.
- `latent_dim`: Bottleneck latent dimension $Z$.
- `sparsity_type`: Sparsity formulation (`"l1"` or `"kl"`).
- `sparsity_weight`: Regularization weight $\lambda$.
- `target_sparsity`: Target activation level $\rho$ (for KL divergence).

No particular window size $W$ is hard-coded into the model definition.
