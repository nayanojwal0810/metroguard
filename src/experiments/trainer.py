"""Leakage-safe training loop for Sparse Autoencoder baseline models."""

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.sae import SparseAutoencoder


@dataclass
class EpochMetrics:
    """Loss metrics recorded for a single training epoch."""

    epoch: int
    train_total_loss: float
    train_recon_loss: float
    train_sparsity_loss: float
    val_total_loss: Optional[float] = None
    val_recon_loss: Optional[float] = None


@dataclass
class TrainingHistory:
    """Complete history of an SAE training run."""

    epochs_trained: int
    history: List[EpochMetrics] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    early_stopped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to serializable dictionary."""
        return {
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "early_stopped": self.early_stopped,
            "final_train_total_loss": (
                self.history[-1].train_total_loss if self.history else None
            ),
            "final_train_recon_loss": (
                self.history[-1].train_recon_loss if self.history else None
            ),
            "final_train_sparsity_loss": (
                self.history[-1].train_sparsity_loss if self.history else None
            ),
            "final_val_total_loss": (
                self.history[-1].val_total_loss if self.history else None
            ),
        }


def set_seed(seed: int) -> None:
    """Configure pseudo-random seeds for repeatable execution across libraries.

    Repeatability & Determinism Scope:
    - Guarantees pseudo-random repeatable execution for CPU computations with identical software builds.
    - If CUDA is available, sets cudnn.deterministic=True and cudnn.benchmark=False.
    - Exact bitwise determinism across different hardware platforms, CUDA driver versions,
      or non-deterministic GPU kernel atomic operations is not guaranteed by seeding alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SAETrainer:
    """Encapsulates model training strictly on training data with deterministic execution."""

    def __init__(
        self,
        model: SparseAutoencoder,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.seed = seed
        set_seed(seed)

        self.device = torch.device(device)
        self.model = model.to(self.device)

        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch over provided dataloader."""
        self.model.train()
        total_loss_accum = 0.0
        recon_loss_accum = 0.0
        sparsity_loss_accum = 0.0
        n_batches = 0

        for batch in dataloader:
            # Handle tuple from TensorDataset or raw tensor
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss_result = self.model.compute_loss(x, output)

            loss_result.total_loss.backward()
            self.optimizer.step()

            total_loss_accum += float(loss_result.total_loss.item())
            recon_loss_accum += float(loss_result.recon_loss.item())
            sparsity_loss_accum += float(loss_result.sparsity_loss.item())
            n_batches += 1

        if n_batches == 0:
            return {"total_loss": 0.0, "recon_loss": 0.0, "sparsity_loss": 0.0}

        return {
            "total_loss": total_loss_accum / n_batches,
            "recon_loss": recon_loss_accum / n_batches,
            "sparsity_loss": sparsity_loss_accum / n_batches,
        }

    def evaluate_loss(self, dataloader: DataLoader) -> Dict[str, float]:
        """Compute average loss metrics in evaluation mode without updating weights."""
        self.model.eval()
        total_loss_accum = 0.0
        recon_loss_accum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                output = self.model(x)
                loss_result = self.model.compute_loss(x, output)

                total_loss_accum += float(loss_result.total_loss.item())
                recon_loss_accum += float(loss_result.recon_loss.item())
                n_batches += 1

        if n_batches == 0:
            return {"total_loss": 0.0, "recon_loss": 0.0}

        return {
            "total_loss": total_loss_accum / n_batches,
            "recon_loss": recon_loss_accum / n_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        min_delta: float = 1e-5,
        verbose: bool = False,
    ) -> TrainingHistory:
        """Run full multi-epoch training loop with optional early stopping.

        IMPORTANT:
            val_loader MUST be derived exclusively from the TRAIN partition
            (e.g., holdout split from train windows). CALIBRATION data must
            never be passed here.
        """
        history = TrainingHistory(epochs_trained=0)
        best_val_loss = float("inf")
        best_epoch: Optional[int] = None
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics: Dict[str, float] = {}

            if val_loader is not None:
                val_metrics = self.evaluate_loss(val_loader)
                val_loss = val_metrics["total_loss"]

                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Snapshot best model weights on CPU
                    best_state_dict = {
                        k: v.clone().detach().cpu()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

            metric_entry = EpochMetrics(
                epoch=epoch,
                train_total_loss=train_metrics["total_loss"],
                train_recon_loss=train_metrics["recon_loss"],
                train_sparsity_loss=train_metrics["sparsity_loss"],
                val_total_loss=val_metrics.get("total_loss"),
                val_recon_loss=val_metrics.get("recon_loss"),
            )
            history.history.append(metric_entry)
            history.epochs_trained = epoch

            if verbose:
                val_str = (
                    f" | Val Total Loss: {metric_entry.val_total_loss:.6f} "
                    f"(Recon: {metric_entry.val_recon_loss:.6f})"
                    if metric_entry.val_total_loss is not None
                    else ""
                )
                print(
                    f"  Epoch {epoch:2d}/{epochs:2d} | "
                    f"Train Total Loss: {metric_entry.train_total_loss:.6f} "
                    f"(Recon: {metric_entry.train_recon_loss:.6f}, Sparsity: {metric_entry.train_sparsity_loss:.6f})"
                    f"{val_str}"
                )

            # Early stopping check
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                history.early_stopped = True
                if verbose:
                    print(f"  Early stopping triggered at epoch {epoch}")
                break

        # Restore best model state when validation was active
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        history.best_epoch = best_epoch
        history.best_loss = best_val_loss if val_loader is not None else None
        return history
