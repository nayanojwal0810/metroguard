"""Configurable Sparse Autoencoder (SAE) with symmetric MLP architecture and L1/KL sparsity."""

from dataclasses import dataclass
import math
from typing import Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.loss import SAELoss, SAELossResult


@dataclass(frozen=True)
class SAEForwardOutput:
    """Outputs produced by Sparse Autoencoder forward pass."""

    reconstruction: torch.Tensor  # Shape: (B, D)
    latent: torch.Tensor  # Shape: (B, Z)
    sample_losses: torch.Tensor  # Shape: (B,) — pointwise MSE error per observation


def get_baseline_dimensions(input_dim: int) -> Tuple[int, int]:
    """Calculate canonical hidden and latent dimensions for a given input dimension.

    Applies the pyramidal compression rule:
        H = min(512, max(32, 2 ** floor(log2(D * 0.65))))
        Z = min(128, max(16, 2 ** floor(log2(H * 0.35))))

    Canonical mappings for 7 analogue sensors across candidate window sizes W:
        - W = 6   (D = 42)   -> H = 32,  Z = 16  (compression: 2.63x)
        - W = 30  (D = 210)  -> H = 128, Z = 32  (compression: 6.56x)
        - W = 90  (D = 630)  -> H = 256, Z = 64  (compression: 9.84x)
        - W = 180 (D = 1260) -> H = 512, Z = 128 (compression: 9.84x)
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}")

    h_exp = int(math.floor(math.log2(max(1.0, input_dim * 0.65))))
    h = min(512, max(32, 2**h_exp))

    z_exp = int(math.floor(math.log2(max(1.0, h * 0.35))))
    z = min(128, max(16, 2**z_exp))

    return h, z


class SparseAutoencoder(nn.Module):
    """Symmetric Multilayer Perceptron (MLP) Sparse Autoencoder for time-series anomaly detection.

    Architecture:
        Input (D) -> Encoder Hidden (H, ReLU) -> Latent Bottleneck (Z, a_z)
                  -> Decoder Hidden (H, ReLU) -> Reconstruction Output (D, Linear)

    Bottleneck activation:
        - ReLU when sparsity_type == 'l1'
        - Sigmoid when sparsity_type == 'kl'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        sparsity_type: Literal["l1", "kl"] = "l1",
        sparsity_weight: float = 1e-4,
        target_sparsity: float = 0.05,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")

        # Derive canonical baseline dimensions if not explicitly configured
        if hidden_dim is None or latent_dim is None:
            default_h, default_z = get_baseline_dimensions(input_dim)
            hidden_dim = default_h if hidden_dim is None else hidden_dim
            latent_dim = default_z if latent_dim is None else latent_dim

        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if latent_dim > hidden_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) cannot exceed hidden_dim ({hidden_dim})"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sparsity_type = sparsity_type

        # Encoder layers
        self.encoder_linear1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_relu = nn.ReLU()
        self.encoder_linear2 = nn.Linear(hidden_dim, latent_dim)

        # Bottleneck activation based on sparsity mechanism
        if sparsity_type == "kl":
            self.latent_activation = nn.Sigmoid()
        elif sparsity_type == "l1":
            self.latent_activation = nn.ReLU()
        else:
            raise ValueError(
                f"sparsity_type must be 'l1' or 'kl', got '{sparsity_type}'"
            )

        # Decoder layers (symmetric architecture)
        self.decoder_linear1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_relu = nn.ReLU()
        self.decoder_linear2 = nn.Linear(hidden_dim, input_dim)

        # Unified loss function module
        self.loss_fn = SAELoss(
            sparsity_type=sparsity_type,
            sparsity_weight=sparsity_weight,
            target_sparsity=target_sparsity,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent bottleneck representation."""
        h = self.encoder_relu(self.encoder_linear1(x))
        z = self.latent_activation(self.encoder_linear2(h))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back into input feature space."""
        h = self.decoder_relu(self.decoder_linear1(z))
        x_hat = self.decoder_linear2(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> SAEForwardOutput:
        """Forward pass through encoder and decoder.

        Accepts either:
            - 2D tensor of shape (B, D)
            - 3D tensor of shape (B, W, C) where W * C == D (automatically flattened)

        Returns:
            SAEForwardOutput containing reconstruction, latent, and pointwise MSE errors.
        """
        # Automatically flatten 3D window batch if provided
        if x.dim() == 3:
            b, w, c = x.shape
            if w * c != self.input_dim:
                raise ValueError(
                    f"Expected flattened dimensions W*C to equal input_dim={self.input_dim}, "
                    f"got {w}*{c}={w * c}"
                )
            x_flat = x.reshape(b, -1)
        elif x.dim() == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected input feature dimension {self.input_dim}, got {x.shape[1]}"
                )
            x_flat = x
        else:
            raise ValueError(
                f"Expected 2D or 3D input tensor, got tensor with {x.dim()} dimensions"
            )

        z = self.encode(x_flat)
        x_hat = self.decode(z)
        sample_losses = self.loss_fn.compute_sample_reconstruction_error(x_flat, x_hat)

        return SAEForwardOutput(
            reconstruction=x_hat,
            latent=z,
            sample_losses=sample_losses,
        )

    def compute_loss(
        self,
        x: torch.Tensor,
        output: SAEForwardOutput,
    ) -> SAELossResult:
        """Calculate total loss, reconstruction MSE, and sparsity penalty."""
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        return self.loss_fn(x, output.reconstruction, output.latent)
