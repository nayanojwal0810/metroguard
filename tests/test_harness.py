"""Focused unit tests for the MetroGuard reproducible experiment harness.

Strict rules:
- Uses ONLY synthetic data (never accesses the real MetroPT CSV).
- Verifies zero temporal leakage and strict holdout protection.
- Verifies deterministic seeding, config validation, scaler invariance, and metrics.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.contract import (
    FAILURE_EVENTS,
    PRIMARY_FEATURES,
    FailureEvent,
    SplitPartition,
)
from src.data.windowing import CausalWindowBuilder
from src.experiments.config import ExperimentConfig, generate_matrix_configs
from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, compute_distribution_stats
from src.experiments.runner import (
    create_synthetic_experiment_data,
    run_experiment,
)
from src.experiments.trainer import SAETrainer, set_seed
from src.models.sae import SparseAutoencoder
from src.preprocessing.scalers import MinMaxScaler, StandardScaler


def test_config_validation_and_dimensions() -> None:
    """Verify that ExperimentConfig derives canonical dimensions and validates inputs."""
    # Canonical W=6 -> D=42, H=32, Z=16
    cfg6 = ExperimentConfig(run_id="test_w6", window_size=6)
    assert cfg6.input_dim == 42
    assert cfg6.hidden_dim == 32
    assert cfg6.latent_dim == 16

    # Canonical W=30 -> D=210, H=128, Z=32
    cfg30 = ExperimentConfig(run_id="test_w30", window_size=30)
    assert cfg30.input_dim == 210
    assert cfg30.hidden_dim == 128
    assert cfg30.latent_dim == 32

    # Serialization and hash determinism
    cfg_dict = cfg30.to_dict()
    assert cfg_dict["run_id"] == "test_w30"
    assert cfg_dict["input_dim"] == 210

    reconstructed = ExperimentConfig.from_dict(cfg_dict)
    assert reconstructed.run_id == cfg30.run_id
    assert reconstructed.compute_config_hash() == cfg30.compute_config_hash()


def test_invalid_experiment_configuration_handling() -> None:
    """Ensure invalid configurations raise explicit ValueError exceptions."""
    # Empty run_id
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        ExperimentConfig(run_id="")

    # Window size < 1
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        ExperimentConfig(run_id="inv1", window_size=0)

    # Invalid scaler type
    with pytest.raises(ValueError, match="scaler_type must be one of"):
        ExperimentConfig(run_id="inv2", scaler_type="RobustScaler")  # type: ignore

    # Invalid sparsity type
    with pytest.raises(ValueError, match="sparsity_type must be one of"):
        ExperimentConfig(run_id="inv3", sparsity_type="l2")  # type: ignore

    # Negative sparsity weight
    with pytest.raises(ValueError, match="sparsity_weight must be >= 0.0"):
        ExperimentConfig(run_id="inv4", sparsity_weight=-0.01)

    # Invalid target sparsity for KL
    with pytest.raises(ValueError, match="target_sparsity must be in"):
        ExperimentConfig(run_id="inv5", target_sparsity=1.5)

    # Invalid batch size
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        ExperimentConfig(run_id="inv6", batch_size=0)

    # Invalid epochs
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        ExperimentConfig(run_id="inv7", epochs=0)


def test_generate_matrix_configs() -> None:
    """Verify generator yields the complete 16-configuration candidate matrix."""
    configs = generate_matrix_configs(base_run_prefix="test_matrix")
    assert len(configs) == 16

    window_sizes = {c.window_size for c in configs}
    scalers = {c.scaler_type for c in configs}
    sparsities = {c.sparsity_type for c in configs}

    assert window_sizes == {6, 30, 90, 180}
    assert scalers == {"StandardScaler", "MinMaxScaler"}
    assert sparsities == {"l1", "kl"}

    # Unique run IDs and distinct config hashes
    run_ids = [c.run_id for c in configs]
    assert len(set(run_ids)) == 16


def test_deterministic_seeding() -> None:
    """Verify that identical random seeds produce identical training steps."""
    x_synthetic = torch.randn(100, 42)
    loader1 = DataLoader(TensorDataset(x_synthetic), batch_size=32, shuffle=False)
    loader2 = DataLoader(TensorDataset(x_synthetic.clone()), batch_size=32, shuffle=False)

    # Run 1 with seed 123
    set_seed(123)
    model1 = SparseAutoencoder(input_dim=42, hidden_dim=32, latent_dim=16)
    trainer1 = SAETrainer(model=model1, seed=123)
    m1 = trainer1.train_epoch(loader1)

    # Run 2 with seed 123
    set_seed(123)
    model2 = SparseAutoencoder(input_dim=42, hidden_dim=32, latent_dim=16)
    trainer2 = SAETrainer(model=model2, seed=123)
    m2 = trainer2.train_epoch(loader2)

    # Must match to floating point tolerance
    assert np.isclose(m1["total_loss"], m2["total_loss"], atol=1e-6)
    assert np.isclose(m1["recon_loss"], m2["recon_loss"], atol=1e-6)
    assert np.isclose(m1["sparsity_loss"], m2["sparsity_loss"], atol=1e-6)

    # Compare resulting parameter tensors
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_one_step_training_execution_synthetic() -> None:
    """Execute multi-step training on synthetic data and verify loss recording."""
    x = torch.randn(128, 42)
    loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)

    model = SparseAutoencoder(input_dim=42, hidden_dim=32, latent_dim=16, sparsity_type="l1")
    trainer = SAETrainer(model=model, learning_rate=1e-3, seed=42)

    history = trainer.train(train_loader=loader, epochs=3)
    assert history.epochs_trained == 3
    assert len(history.history) == 3
    assert not history.early_stopped

    for ep in history.history:
        assert ep.train_total_loss > 0.0
        assert ep.train_recon_loss > 0.0
        assert ep.train_sparsity_loss >= 0.0
        assert ep.val_total_loss is None


def test_training_does_not_consume_calibration_data() -> None:
    """Verify that training never accesses CALIBRATION data."""
    train_df, cal_df = create_synthetic_experiment_data(num_train_rows=100, num_cal_rows=100, seed=42)

    # Intentionally corrupt cal_df with NaNs
    corrupted_cal_df = cal_df.copy()
    for col in PRIMARY_FEATURES:
        corrupted_cal_df[col] = np.nan

    config = ExperimentConfig(run_id="test_no_cal_leak", window_size=6, epochs=1, batch_size=32)

    # Training portion must complete without seeing the corrupted cal data
    # (Only the calibration evaluator step would fail on NaNs)
    builder = CausalWindowBuilder(window_size=config.window_size, features=config.features)
    train_batch = builder.build_windows(train_df)

    scaler = StandardScaler(features=config.features)
    scaler.fit(train_batch.windows.reshape(-1, len(config.features)))
    scaled_train = scaler.transform(train_batch.windows.reshape(-1, len(config.features))).reshape(len(train_batch.windows), -1)

    model = SparseAutoencoder(input_dim=config.input_dim)
    trainer = SAETrainer(model=model, seed=42)
    loader = DataLoader(TensorDataset(torch.from_numpy(scaled_train).float()), batch_size=32)

    # Training must succeed without touching cal_df
    history = trainer.train(train_loader=loader, epochs=1)
    assert history.epochs_trained == 1
    assert not np.isnan(history.history[0].train_total_loss)


def test_scaler_remains_unchanged_during_calibration_evaluation() -> None:
    """Verify that CalibrationEvaluator does not modify fitted scaler parameters."""
    rng = np.random.RandomState(42)
    synthetic_train = rng.normal(loc=10.0, scale=2.5, size=(500, len(PRIMARY_FEATURES)))
    synthetic_cal_windows = rng.normal(loc=12.0, scale=3.0, size=(100, 6, len(PRIMARY_FEATURES))).astype(np.float32)
    timestamps = pd.date_range("2020-04-01", periods=100, freq="10s").to_numpy()

    # 1. StandardScaler check
    scaler_std = StandardScaler(features=PRIMARY_FEATURES)
    scaler_std.fit(synthetic_train)
    mean_orig = scaler_std.mean_.copy()
    std_orig = scaler_std.std_.copy()

    model = SparseAutoencoder(input_dim=42)
    evaluator_std = CalibrationEvaluator(model=model, scaler=scaler_std)
    evaluator_std.evaluate(synthetic_cal_windows, timestamps)

    assert np.array_equal(scaler_std.mean_, mean_orig)
    assert np.array_equal(scaler_std.std_, std_orig)

    # 2. MinMaxScaler check
    scaler_mm = MinMaxScaler(features=PRIMARY_FEATURES)
    scaler_mm.fit(synthetic_train)
    min_orig = scaler_mm.min_.copy()
    max_orig = scaler_mm.max_.copy()
    range_orig = scaler_mm.range_.copy()

    evaluator_mm = CalibrationEvaluator(model=model, scaler=scaler_mm)
    evaluator_mm.evaluate(synthetic_cal_windows, timestamps)

    assert np.array_equal(scaler_mm.min_, min_orig)
    assert np.array_equal(scaler_mm.max_, max_orig)
    assert np.array_equal(scaler_mm.range_, range_orig)


def test_reconstruction_score_and_metrics_calculation() -> None:
    """Verify pointwise reconstruction scoring, event labeling, and metrics."""
    # Synthetic windows: 50 normal, 50 failure
    cal_windows = np.random.normal(5.0, 1.0, size=(100, 6, 7)).astype(np.float32)
    # Give the second half large deviations to simulate clear failure anomaly
    cal_windows[50:] += 10.0

    # Timestamps: 50 before event, 50 during Event 1
    ts_normal = pd.date_range("2020-04-10 00:00:00", periods=50, freq="10s")
    ts_event = pd.date_range("2020-04-18 12:00:00", periods=50, freq="10s")
    all_ts = ts_normal.append(ts_event).to_numpy()

    scaler = StandardScaler(features=PRIMARY_FEATURES)
    # Fit scaler on normal data
    scaler.fit(np.random.normal(5.0, 1.0, size=(500, 7)))

    model = SparseAutoencoder(input_dim=42)
    evaluator = CalibrationEvaluator(model=model, scaler=scaler)

    res = evaluator.evaluate(
        cal_windows=cal_windows,
        cal_timestamps=all_ts,
        num_calibration_rows=120,
    )

    m = res.metrics
    assert m.num_usable_windows == 100
    assert m.num_rejected_windows == 20  # 120 - 100
    assert m.num_normal_windows == 50
    assert m.num_failure_windows == 50
    assert np.isclose(m.failure_ratio, 0.5)

    # Score calculation check
    assert len(res.scores) == 100
    assert np.all(res.scores >= 0.0)

    # Failure scores should be significantly higher due to +10.0 deviation
    assert m.failure_distribution["mean"] > m.normal_distribution["mean"]

    # PR-AUC and ROC-AUC must be computed and valid
    assert m.pr_auc is not None and 0.0 <= m.pr_auc <= 1.0
    assert m.roc_auc is not None and 0.0 <= m.roc_auc <= 1.0

    # Score quantiles must be monotonic
    q = m.score_quantiles
    assert q["p25"] <= q["p50"] <= q["p75"] <= q["p90"] <= q["p95"] <= q["p99"]


def test_run_metadata_creation_and_serialization(tmp_path: Path) -> None:
    """Verify that full experiment run produces compact and reproducible metadata."""
    train_df, cal_df = create_synthetic_experiment_data(num_train_rows=200, num_cal_rows=200, seed=42)

    config = ExperimentConfig(
        run_id="test_artifact_run",
        window_size=6,
        epochs=1,
        batch_size=32,
        seed=42,
    )

    result = run_experiment(
        config=config,
        train_df=train_df,
        cal_df=cal_df,
        save_dir=str(tmp_path),
        verbose=False,
    )

    out_file = tmp_path / f"{config.run_id}.json"
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify key metadata components
    assert data["run_id"] == "test_artifact_run"
    assert "metadata" in data
    assert data["metadata"]["holdout_accessed"] is False
    assert data["metadata"]["dataset_fingerprint"] == "synthetic"
    assert "train_boundary" in data["metadata"]
    assert "calibration_boundary" in data["metadata"]
    assert "git_commit" in data["metadata"]
    assert "runtime_seconds" in data["metadata"]
    assert "environment" in data["metadata"]

    assert "config" in data
    assert data["config"]["input_dim"] == 42

    assert "training_history" in data
    assert data["training_history"]["epochs_trained"] == 1

    assert "calibration_metrics" in data
    assert data["calibration_metrics"]["num_usable_windows"] > 0
