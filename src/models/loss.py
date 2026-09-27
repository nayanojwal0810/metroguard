"""Loss functions for Sparse Autoencoder (Reconstruction MSE + L1/KL sparsity)."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional
import torch
import torch.nn as nn


@dataclass(frozen=True)
class SAELossResult:
    """Breakdown of Sparse Autoencoder loss terms."""

    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    weighted_sparsity_loss: torch.Tensor


class SAELoss(nn.Module):
    """Computes total SAE training loss = reconstruction MSE + weighted sparsity penalty.

    Supports candidate sparsity mechanisms:
    - L1 activity penalty on latent activations: lambda * mean(|z|)
    - KL divergence penalty against target firing probability rho: lambda * KL(rho || rho_hat)
    """

    def __init__(
        self,
        sparsity_type: Literal["l1", "kl"] = "l1",
        sparsity_weight: float = 1e-4,
        target_sparsity: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        valid_types = ("l1", "kl")
        if sparsity_type not in valid_types:
            raise ValueError(f"sparsity_type must be one of {valid_types}, got '{sparsity_type}'")
        if sparsity_weight < 0.0:
            raise ValueError(f"sparsity_weight must be >= 0.0, got {sparsity_weight}")
        if target_sparsity <= 0.0 or target_sparsity >= 1.0:
            raise ValueError(
                f"target_sparsity must be in (0.0, 1.0), got {target_sparsity}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {eps}")

        self.sparsity_type = sparsity_type
        self.sparsity_weight = float(sparsity_weight)
        self.target_sparsity = float(target_sparsity)
        self.eps = float(eps)

    def compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-averaged Mean Squared Error over all feature dimensions."""
        return torch.mean((x - x_hat) ** 2)

    def compute_sample_reconstruction_error(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Pointwise MSE reconstruction error per observation in batch.

        Returns 1D tensor of shape (B,).
        """
        # If input has more than 2 dimensions, flatten trailing dimensions
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
            x_hat = x_hat.reshape(x_hat.shape[0], -1)
        return torch.mean((x - x_hat) ** 2, dim=-1)

    def compute_l1_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Computes mean absolute activation across batch and latent units."""
        return torch.mean(torch.abs(z))

    def compute_kl_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Computes average KL divergence between target rho and batch mean activation rho_hat.

        KL(rho || rho_hat) = rho * log(rho / rho_hat) + (1 - rho) * log((1 - rho) / (1 - rho_hat))
        """
        # Batch average activation per latent unit: shape (Z,)
        rho_hat = torch.mean(z, dim=0)
        # Numerical guard: clamp rho_hat to [eps, 1 - eps]
        rho_hat = torch.clamp(rho_hat, min=self.eps, max=1.0 - self.eps)
        rho = torch.tensor(self.target_sparsity, device=z.device, dtype=z.dtype)

        # Pointwise KL divergence per latent unit
        kl_div = (
            rho * torch.log(rho / rho_hat)
            + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat))
        )
        # Average across all latent units
        return torch.mean(kl_div)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
    ) -> SAELossResult:
        """Calculate total loss, reconstruction loss, and sparsity loss.

        Args:
            x: Original input tensor of shape (B, D).
            x_hat: Reconstructed tensor of shape (B, D).
            z: Latent bottleneck activation tensor of shape (B, Z).

        Returns:
            SAELossResult container with individual loss components.
        """
        recon_loss = self.compute_reconstruction_loss(x, x_hat)

        if self.sparsity_weight == 0.0:
            sparsity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        elif self.sparsity_type == "l1":
            sparsity_loss = self.compute_l1_penalty(z)
        else:
            sparsity_loss = self.compute_kl_penalty(z)

        weighted_sparsity = self.sparsity_weight * sparsity_loss
        total_loss = recon_loss + weighted_sparsity

        return SAELossResult(
            total_loss=total_loss,
            recon_loss=recon_loss,
            sparsity_loss=sparsity_loss,
            weighted_sparsity_loss=weighted_sparsity,
        )
