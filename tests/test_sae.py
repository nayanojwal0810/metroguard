"""Unit tests for Sparse Autoencoder (SAE) model, baseline dimensions, and loss functions."""

import numpy as np
import pytest
import torch

from src.models.loss import SAELoss
from src.models.sae import (
    SparseAutoencoder,
    get_baseline_dimensions,
)


def test_baseline_dimension_rules_across_candidate_windows():
    """Verify that canonical dimension formula maps candidate window sizes to expected (H, Z)."""
    expected_mappings = {
        6: (42, 32, 16),
        30: (210, 128, 32),
        90: (630, 256, 64),
        180: (1260, 512, 128),
    }

    for w, (d, expected_h, expected_z) in expected_mappings.items():
        assert d == 7 * w
        h, z = get_baseline_dimensions(d)
        assert h == expected_h, f"Failed for W={w}, D={d}: expected H={expected_h}, got {h}"
        assert z == expected_z, f"Failed for W={w}, D={d}: expected Z={expected_z}, got {z}"


def test_model_construction_across_all_candidate_windows():
    """Verify instantiation and layer dimensions for W = 6, 30, 90, 180."""
    candidate_windows = (6, 30, 90, 180)

    for w in candidate_windows:
        d = 7 * w
        h, z = get_baseline_dimensions(d)
        model = SparseAutoencoder(input_dim=d)

        assert model.input_dim == d
        assert model.hidden_dim == h
        assert model.latent_dim == z

        # Encoder weights
        assert model.encoder_linear1.weight.shape == (h, d)
        assert model.encoder_linear2.weight.shape == (z, h)
        # Decoder weights
        assert model.decoder_linear1.weight.shape == (h, z)
        assert model.decoder_linear2.weight.shape == (d, h)


def test_forward_pass_2d_and_3d_shapes():
    """Verify forward pass correctly processes both flattened 2D and 3D window batches."""
    batch_size = 4
    w = 6
    n_features = 7
    d = w * n_features
    h, z = get_baseline_dimensions(d)

    model = SparseAutoencoder(input_dim=d)

    # 1. Test 2D input: (B, D)
    x_2d = torch.randn(batch_size, d)
    out_2d = model(x_2d)

    assert out_2d.reconstruction.shape == (batch_size, d)
    assert out_2d.latent.shape == (batch_size, z)
    assert out_2d.sample_losses.shape == (batch_size,)

    # 2. Test 3D input: (B, W, C)
    x_3d = torch.randn(batch_size, w, n_features)
    out_3d = model(x_3d)

    assert out_3d.reconstruction.shape == (batch_size, d)
    assert out_3d.latent.shape == (batch_size, z)
    assert out_3d.sample_losses.shape == (batch_size,)


def test_deterministic_inference():
    """Verify model produces deterministic outputs in eval mode given identical input."""
    model = SparseAutoencoder(input_dim=42)
    model.eval()

    x = torch.randn(5, 42)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1.reconstruction, out2.reconstruction)
    torch.testing.assert_close(out1.latent, out2.latent)
    torch.testing.assert_close(out1.sample_losses, out2.sample_losses)


def test_reconstruction_error_calculation():
    """Verify sample-level MSE reconstruction error calculation against manual formula."""
    model = SparseAutoencoder(input_dim=4, hidden_dim=4, latent_dim=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    x_hat = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    # Row 0 error: ((1-1)^2 + (2-1)^2 + (3-1)^2 + (4-1)^2) / 4 = (0 + 1 + 4 + 9) / 4 = 14 / 4 = 3.5
    # Row 1 error: ((0-1)^2 + (0-1)^2 + (0-1)^2 + (0-1)^2) / 4 = 4 / 4 = 1.0
    expected_sample_losses = torch.tensor([3.5, 1.0])

    loss_fn = model.loss_fn
    sample_losses = loss_fn.compute_sample_reconstruction_error(x, x_hat)
    torch.testing.assert_close(sample_losses, expected_sample_losses)


def test_l1_sparsity_loss():
    """Verify L1 sparsity penalty and total loss calculation."""
    sparsity_weight = 0.01
    model = SparseAutoencoder(
        input_dim=42,
        sparsity_type="l1",
        sparsity_weight=sparsity_weight,
    )

    x = torch.randn(8, 42)
    out = model(x)
    loss_result = model.compute_loss(x, out)

    # ReLU bottleneck must be non-negative
    assert (out.latent >= 0.0).all()

    # Manual L1 penalty
    expected_l1 = torch.mean(torch.abs(out.latent))
    torch.testing.assert_close(loss_result.sparsity_loss, expected_l1)

    expected_total = loss_result.recon_loss + sparsity_weight * expected_l1
    torch.testing.assert_close(loss_result.total_loss, expected_total)


def test_kl_sparsity_loss():
    """Verify KL divergence sparsity penalty with Sigmoid bottleneck."""
    sparsity_weight = 0.05
    target_sparsity = 0.05
    model = SparseAutoencoder(
        input_dim=42,
        sparsity_type="kl",
        sparsity_weight=sparsity_weight,
        target_sparsity=target_sparsity,
    )

    x = torch.randn(10, 42)
    out = model(x)
    loss_result = model.compute_loss(x, out)

    # Sigmoid bottleneck must be strictly within (0, 1)
    assert (out.latent > 0.0).all()
    assert (out.latent < 1.0).all()

    # KL penalty must be non-negative and finite
    assert loss_result.sparsity_loss >= 0.0
    assert not torch.isnan(loss_result.sparsity_loss)
    assert not torch.isinf(loss_result.sparsity_loss)

    expected_total = loss_result.recon_loss + sparsity_weight * loss_result.sparsity_loss
    torch.testing.assert_close(loss_result.total_loss, expected_total)


def test_invalid_configurations_and_dimensions():
    """Verify that invalid parameters and dimensional mismatches raise ValueError."""
    # Invalid input dimension
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        SparseAutoencoder(input_dim=0)

    # Invalid hidden / latent dimensions
    with pytest.raises(ValueError, match="latent_dim .* cannot exceed hidden_dim"):
        SparseAutoencoder(input_dim=42, hidden_dim=16, latent_dim=32)

    # Invalid sparsity type
    with pytest.raises(ValueError, match="sparsity_type must be"):
        SparseAutoencoder(input_dim=42, sparsity_type="invalid_type")

    # Invalid sparsity weight
    with pytest.raises(ValueError, match="sparsity_weight must be >= 0"):
        SAELoss(sparsity_weight=-0.1)

    # Invalid target sparsity for KL
    with pytest.raises(ValueError, match="target_sparsity must be in"):
        SAELoss(target_sparsity=1.5)

    # Dimensional mismatch on forward pass
    model = SparseAutoencoder(input_dim=42)
    with pytest.raises(ValueError, match="Expected input feature dimension 42"):
        model(torch.randn(4, 20))
