# MetroGuard Reproducible Experiment Harness Specification

## 1. Overview and Architecture

The experiment harness provides a reproducible, leakage-safe pipeline to train configurable Sparse Autoencoder (SAE) models on chronological `TRAIN` data and evaluate them on `CALIBRATION` data without touching `FINAL HOLDOUT`.

```
Raw CSV / Preprocessed Partition
          │
          ▼
┌──────────────────┐
│ Schema & Split   │  --> Drops HOLDOUT (>= 2020-06-01) completely
│ Validation       │
└─────────┬────────┘
          │
          ├──> TRAIN Partition (2020-02-01 to 2020-03-31)
          │         │
          │         ▼
          │    Causal Window Builder (W in {6, 30, 90, 180}, gap rule delta_t > 60s)
          │         │
          │         ▼
          │    Train Scaler Fit (StandardScaler / MinMaxScaler fit strictly on train)
          │         │
          │         ▼
          │    Train Scaler Transform
          │         │
          │         ▼
          │    SAE Model Training (PyTorch Adam/SGD, L1/KL loss, seeded)
          │         │
          │         ▼
          └──> CALIBRATION Partition (2020-04-01 to 2020-05-31)
                    │
                    ▼
               Causal Window Builder
                    │
                    ▼
               Calibration Transform (using fitted TRAIN scaler; parameters frozen)
                    │
                    ▼
               Calibration Evaluator (threshold-free metrics, event labeling, PR-AUC, ROC-AUC)
                    │
                    ▼
               Run Metadata & Artifact JSON
```

---

## 2. Configuration Schema (`ExperimentConfig`)

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `run_id` | `str` | *required* | Unique identifier for the experiment run |
| `window_size` | `int` | `30` | Observation window size $W \in \{6, 30, 90, 180\}$ |
| `stride` | `int` | `1` | Sliding step across valid segments |
| `scaler_type` | `str` | `"StandardScaler"` | Scaler variant: `"StandardScaler"` or `"MinMaxScaler"` |
| `sparsity_type` | `str` | `"l1"` | Regularization mechanism: `"l1"` (activity) or `"kl"` (divergence) |
| `sparsity_weight` | `float` | `1e-4` | Weight $\lambda$ balancing reconstruction loss vs sparsity |
| `target_sparsity` | `float` | `0.05` | Target firing rate $\rho$ for KL divergence penalty |
| `hidden_dim` | `Optional[int]` | `None` | Hidden layer dimension $H$ (auto-derived via canonical rule if omitted) |
| `latent_dim` | `Optional[int]` | `None` | Latent layer dimension $Z$ (auto-derived via canonical rule if omitted) |
| `seed` | `int` | `42` | Random seed for deterministic weight init and batch shuffling |
| `optimizer` | `str` | `"adam"` | Optimizer name: `"adam"`, `"adamw"`, or `"sgd"` |
| `learning_rate` | `float` | `1e-3` | Initial learning rate |
| `batch_size` | `int` | `256` | Mini-batch size |
| `epochs` | `int` | `10` | Maximum training epochs |
| `early_stopping_patience` | `Optional[int]` | `None` | Epochs without improvement before early termination |
| `validation_fraction` | `float` | `0.1` | Fraction of chronological train data reserved for validation |
| `device` | `str` | `"cpu"` | Computation device (`"cpu"` or `"cuda"`) |
| `features` | `Tuple[str, ...]` | Canonical 7 | Analogue sensor features |

### Canonical Dimension Derivation
For 7 input features:
- $W = 6 \implies D = 42 \implies H = 32, Z = 16$
- $W = 30 \implies D = 210 \implies H = 128, Z = 32$
- $W = 90 \implies D = 630 \implies H = 256, Z = 64$
- $W = 180 \implies D = 1260 \implies H = 512, Z = 128$

---

## 3. Calibration Evaluation Metrics

Evaluation on the calibration partition is strictly threshold-free:
- **Sample Reconstruction Error**: $s_t = \frac{1}{D} \sum_{d=1}^D (x_d - \hat{x}_d)^2$
- **Distributions**: Mean, standard deviation, median, min, max, 25th, 75th, 90th, 95th, 99th percentiles for normal periods vs failure periods.
- **Documented Event Labeling**:
  - `Event_1`: `2020-04-18 00:00:00` to `2020-04-18 23:59:59`
  - `Event_2`: `2020-05-29 23:30:00` to `2020-05-30 06:00:00`
- **Ranking Metrics**: Precision-Recall Area Under Curve (PR-AUC) and ROC-AUC (descriptive, threshold-free).
- **Accounting**: Usable window count vs rejected windows (insufficient history or service gaps).

---

## 4. Controlled Candidate Matrix (16 Configurations)

The full baseline evaluation matrix spans:
- Window sizes $W \in \{6, 30, 90, 180\}$ (4 values)
- Scalers $\in \{\text{StandardScaler}, \text{MinMaxScaler}\}$ (2 values)
- Sparsity $\in \{\text{L1}, \text{KL}\}$ (2 values)
- Total: $4 \times 2 \times 2 = 16$ candidate configurations.

---

## 5. Usage Commands

```bash
# Run lightweight synthetic smoke test (< 5 seconds on CPU)
python -m src.experiments.runner --smoke

# Inspect the 16 candidate configurations
python -m src.experiments.runner --list-matrix

# View estimated compute and memory requirements
python -m src.experiments.runner --estimate-resources
```
