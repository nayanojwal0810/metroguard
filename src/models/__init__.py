"""Sparse Autoencoder models and loss modules for MetroGuard."""

from src.models.loss import SAELoss, SAELossResult
from src.models.sae import (
    SAEForwardOutput,
    SparseAutoencoder,
    get_baseline_dimensions,
)

__all__ = [
    "SAELoss",
    "SAELossResult",
    "SAEForwardOutput",
    "SparseAutoencoder",
    "get_baseline_dimensions",
]
