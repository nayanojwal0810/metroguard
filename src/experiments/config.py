"""Reproducible experiment configuration for Sparse Autoencoder baseline training."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.data.contract import PRIMARY_FEATURES, WINDOW_CANDIDATES
from src.models.sae import get_baseline_dimensions


@dataclass
class ExperimentConfig:
    """Configuration specification for a controlled SAE training and calibration run."""

    run_id: str
    window_size: int = 30
    stride: int = 1
    scaler_type: Literal["StandardScaler", "MinMaxScaler"] = "StandardScaler"
    sparsity_type: Literal["l1", "kl"] = "l1"
    sparsity_weight: float = 1e-4
    target_sparsity: float = 0.05
    hidden_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    seed: int = 42
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    early_stopping_patience: Optional[int] = None
    validation_fraction: float = 0.1
    device: str = "cpu"
    features: Tuple[str, ...] = field(default_factory=lambda: PRIMARY_FEATURES)

    def __post_init__(self) -> None:
        self.validate()
        # Derive canonical dimensions if not explicitly overridden
        d = len(self.features) * self.window_size
        default_h, default_z = get_baseline_dimensions(d)
        if self.hidden_dim is None:
            self.hidden_dim = default_h
        if self.latent_dim is None:
            self.latent_dim = default_z

    @property
    def input_dim(self) -> int:
        """Flattened input feature dimension D = num_features * window_size."""
        return len(self.features) * self.window_size

    def validate(self) -> None:
        """Validate that configuration settings adhere to project standards."""
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string.")

        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        valid_scalers = ("StandardScaler", "MinMaxScaler")
        if self.scaler_type not in valid_scalers:
            raise ValueError(
                f"scaler_type must be one of {valid_scalers}, got '{self.scaler_type}'"
            )

        valid_sparsity = ("l1", "kl")
        if self.sparsity_type not in valid_sparsity:
            raise ValueError(
                f"sparsity_type must be one of {valid_sparsity}, got '{self.sparsity_type}'"
            )

        if self.sparsity_weight < 0.0:
            raise ValueError(f"sparsity_weight must be >= 0.0, got {self.sparsity_weight}")

        if self.target_sparsity <= 0.0 or self.target_sparsity >= 1.0:
            raise ValueError(
                f"target_sparsity must be in (0.0, 1.0), got {self.target_sparsity}"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0.0, got {self.learning_rate}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}"
            )

        if self.validation_fraction < 0.0 or self.validation_fraction >= 0.5:
            raise ValueError(
                f"validation_fraction must be in [0.0, 0.5), got {self.validation_fraction}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        d["input_dim"] = self.input_dim
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Reconstruct configuration from dictionary."""
        clean_data = {k: v for k, v in data.items() if k != "input_dim"}
        if "features" in clean_data and isinstance(clean_data["features"], list):
            clean_data["features"] = tuple(clean_data["features"])
        return cls(**clean_data)

    def compute_config_hash(self) -> str:
        """Produce deterministic 8-character hash of configuration settings."""
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:8]


def generate_matrix_configs(
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 256,
) -> List[ExperimentConfig]:
    """Generate the full 16 candidate baseline experiment configurations.

    Matrix:
        W in {6, 30, 90, 180} (4 values)
        scaler in {StandardScaler, MinMaxScaler} (2 values)
        sparsity in {l1, kl} (2 values)
        Total: 4 * 2 * 2 = 16 configurations.
    """
    configs: List[ExperimentConfig] = []

    for w in WINDOW_CANDIDATES:
        for scaler in ("StandardScaler", "MinMaxScaler"):
            for sparsity in ("l1", "kl"):
                # Documented baseline default regularization weights
                weight = 1e-4 if sparsity == "l1" else 1e-2
                scaler_short = "standard" if scaler == "StandardScaler" else "minmax"
                run_id = f"w{w:02d}_{scaler_short}_{sparsity}_s{seed}"

                cfg = ExperimentConfig(
                    run_id=run_id,
                    window_size=w,
                    stride=1,
                    scaler_type=scaler,
                    sparsity_type=sparsity,
                    sparsity_weight=weight,
                    target_sparsity=0.05,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                configs.append(cfg)

    return configs
