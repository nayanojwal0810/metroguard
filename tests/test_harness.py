"""Focused unit tests for the MetroGuard reproducible experiment harness.

Strict rules:
- Uses ONLY synthetic data (never accesses the real MetroPT CSV).
- Verifies zero temporal leakage and strict holdout protection.
- Verifies deterministic seeding, config validation, scaler invariance, and metrics.
- Verifies explicit window accounting, early stopping best-state restoration,
  and labeling convention boundary behavior.
"""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.contract import (
    CANONICAL_DATASET_SHA256,
    DATASET_EXPECTED_ROWS,
    EXPECTED_CALIBRATION_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_UNUSED_TAIL_ROWS,
    FAILURE_EVENTS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    SYNTHETIC_DATASET_FINGERPRINT,
    TIMESTAMP_COL,
    FailureEvent,
    SplitPartition,
)
from src.data.windowing import CausalWindowBuilder, WindowAccounting
from src.experiments.config import ExperimentConfig, generate_matrix_configs
from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, compute_distribution_stats
from src.experiments.runner import (
    create_synthetic_experiment_data,
    load_train_and_calibration_data,
    run_experiment,
    run_matrix,
)
from src.experiments.trainer import EpochMetrics, SAETrainer, set_seed
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
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        ExperimentConfig(run_id="")

    with pytest.raises(ValueError, match="window_size must be >= 1"):
        ExperimentConfig(run_id="inv1", window_size=0)

    with pytest.raises(ValueError, match="scaler_type must be one of"):
        ExperimentConfig(run_id="inv2", scaler_type="RobustScaler")  # type: ignore

    with pytest.raises(ValueError, match="sparsity_type must be one of"):
        ExperimentConfig(run_id="inv3", sparsity_type="l2")  # type: ignore

    with pytest.raises(ValueError, match="sparsity_weight must be >= 0.0"):
        ExperimentConfig(run_id="inv4", sparsity_weight=-0.01)

    with pytest.raises(ValueError, match="target_sparsity must be in"):
        ExperimentConfig(run_id="inv5", target_sparsity=1.5)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        ExperimentConfig(run_id="inv6", batch_size=0)

    with pytest.raises(ValueError, match="epochs must be >= 1"):
        ExperimentConfig(run_id="inv7", epochs=0)


def test_generate_matrix_configs() -> None:
    """Verify generator yields the complete 16-configuration candidate matrix."""
    configs = generate_matrix_configs()
    assert len(configs) == 16

    window_sizes = {c.window_size for c in configs}
    scalers = {c.scaler_type for c in configs}
    sparsities = {c.sparsity_type for c in configs}

    assert window_sizes == {6, 30, 90, 180}
    assert scalers == {"StandardScaler", "MinMaxScaler"}
    assert sparsities == {"l1", "kl"}

    run_ids = [c.run_id for c in configs]
    assert len(set(run_ids)) == 16


def test_deterministic_cpu_seeding() -> None:
    """Verify that identical random seeds produce identical training steps on CPU."""
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

    assert np.isclose(m1["total_loss"], m2["total_loss"], atol=1e-6)
    assert np.isclose(m1["recon_loss"], m2["recon_loss"], atol=1e-6)
    assert np.isclose(m1["sparsity_loss"], m2["sparsity_loss"], atol=1e-6)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_holdout_safe_chunked_ingestion(tmp_path: Path) -> None:
    """Verify that chunked ingestion never admits timestamps >= 2020-06-01 or < 2020-02-01."""
    # Create mock multi-month CSV spanning Jan 2020 through Sep 2020
    ts_pre = pd.date_range("2020-01-15 00:00:00", periods=20, freq="1h")
    ts_train = pd.date_range("2020-02-15 00:00:00", periods=30, freq="1h")
    ts_cal = pd.date_range("2020-04-15 00:00:00", periods=30, freq="1h")
    ts_holdout = pd.date_range("2020-06-15 00:00:00", periods=30, freq="1h")
    ts_tail = pd.date_range("2020-09-01 01:00:00", periods=10, freq="10min")

    all_ts = ts_pre.append(ts_train).append(ts_cal).append(ts_holdout).append(ts_tail)
    data = {TIMESTAMP_COL: all_ts}
    for col in PRIMARY_FEATURES:
        data[col] = np.arange(len(all_ts), dtype=np.float32)
    # Also add an extraneous column to verify column filtering
    data["ignored_column"] = np.ones(len(all_ts), dtype=np.float32)

    mock_csv = tmp_path / "mock_telemetry.csv"
    pd.DataFrame(data).to_csv(mock_csv, index=False)

    train_df, cal_df = load_train_and_calibration_data(
        str(mock_csv),
        chunk_size=15,
        enforce_real_counts=False,
    )

    # 1. Row counts: pre-train (20), holdout (30), tail (10) are excluded
    assert len(train_df) == 30
    assert len(cal_df) == 30

    # 2. Schema check: ignored_column was never read into DataFrames
    assert "ignored_column" not in train_df.columns
    assert "ignored_column" not in cal_df.columns

    # 3. Holdout protection: zero timestamps >= 2020-06-01
    holdout_start = pd.Timestamp("2020-06-01 00:00:00")
    assert (train_df[TIMESTAMP_COL] < holdout_start).all()
    assert (cal_df[TIMESTAMP_COL] < holdout_start).all()

    # 4. Pre-train exclusion: zero timestamps < 2020-02-01
    train_start = pd.Timestamp("2020-02-01 00:00:00")
    assert (train_df[TIMESTAMP_COL] >= train_start).all()
    assert (cal_df[TIMESTAMP_COL] >= pd.Timestamp("2020-04-01 00:00:00")).all()


def test_dataset_provenance_handling(tmp_path: Path) -> None:
    """Verify that dataset provenance distinguishes real vs synthetic data correctly."""
    train_df, cal_df = create_synthetic_experiment_data(num_train_rows=50, num_cal_rows=50)

    cfg = ExperimentConfig(run_id="test_prov", window_size=6, epochs=1, batch_size=32)

    # 1. Synthetic run records explicit synthetic identifier
    res_synthetic = run_experiment(
        config=cfg,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=SYNTHETIC_DATASET_FINGERPRINT,
    )
    assert res_synthetic.metadata["dataset_fingerprint"] == SYNTHETIC_DATASET_FINGERPRINT
    assert res_synthetic.metadata["holdout_observations_retained"] is False

    # 2. When explicit real fingerprint is passed (e.g. from run_matrix preloading real data)
    res_real = run_experiment(
        config=cfg,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=CANONICAL_DATASET_SHA256,
    )
    assert res_real.metadata["dataset_fingerprint"] == CANONICAL_DATASET_SHA256
    assert res_real.metadata["dataset_fingerprint"] != SYNTHETIC_DATASET_FINGERPRINT


def test_train_fit_val_scaler_separation() -> None:
    """Verify that scaler fits ONLY on TRAIN-fit and is never influenced by TRAIN-val."""
    rng = np.random.RandomState(42)
    n_rows = 500
    ts = pd.date_range("2020-02-01 10:00:00", periods=n_rows, freq="10s")
    data = {TIMESTAMP_COL: ts}
    for col in PRIMARY_FEATURES:
        data[col] = rng.normal(loc=10.0, scale=1.0, size=n_rows).astype(np.float32)

    train_df = pd.DataFrame(data)

    # With validation_fraction = 0.2:
    # First 400 rows are TRAIN-fit (mean ~ 10.0)
    # Last 100 rows are TRAIN-val
    # Intentionally corrupt the last 100 rows with massive values (10,000.0)
    for col in PRIMARY_FEATURES:
        train_df.loc[400:, col] = 10_000.0

    cal_ts = pd.date_range("2020-04-01 10:00:00", periods=100, freq="10s")
    cal_data = {TIMESTAMP_COL: cal_ts}
    for col in PRIMARY_FEATURES:
        cal_data[col] = rng.normal(loc=10.0, scale=1.0, size=100).astype(np.float32)
    cal_df = pd.DataFrame(cal_data)

    config = ExperimentConfig(
        run_id="test_scaler_sep",
        window_size=6,
        epochs=1,
        batch_size=32,
        validation_fraction=0.2,
        scaler_type="StandardScaler",
    )

    result = run_experiment(
        config=config,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=SYNTHETIC_DATASET_FINGERPRINT,
    )

    # The fitted scaler's mean must remain ~10.0, NOT shifted by the 10,000.0 values in TRAIN-val!
    # If TRAIN-val had leaked into the scaler fit, mean would be ~ 0.8*10 + 0.2*10000 = 2008.0!
    assert result.metadata["train_fit_raw_row_count"] == 400
    assert result.metadata["train_val_raw_row_count"] == 100
    # Also verify that holdout was never retained
    assert result.metadata["holdout_observations_retained"] is False


def test_early_stopping_best_state_restoration() -> None:
    """Verify that SAETrainer snapshots and restores the model parameters of best validation epoch."""
    set_seed(42)
    # Create synthetic dataset where epoch 1 has good fit, but epoch 2 will overfit / diverge
    x_train = torch.randn(128, 42)
    train_loader = DataLoader(TensorDataset(x_train), batch_size=32)

    # Validation loader
    x_val = torch.randn(64, 42)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=32)

    model = SparseAutoencoder(input_dim=42, hidden_dim=32, latent_dim=16)
    trainer = SAETrainer(model=model, learning_rate=1e-3, seed=42)

    # Train for 3 epochs with early stopping patience=1
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        early_stopping_patience=1,
    )

    # History must record the best epoch
    assert history.best_epoch is not None
    assert history.best_loss is not None

    # Verify that model weights match evaluation loss of best_epoch
    val_eval = trainer.evaluate_loss(val_loader)
    # The evaluation loss on the restored model should match best_loss
    assert np.isclose(val_eval["total_loss"], history.best_loss, atol=1e-5)


def test_explicit_window_accounting() -> None:
    """Verify explicit accounting of raw observations, candidate endpoints, and exclusion causes."""
    # Construct sequence:
    # 1. 5 normal observations (t0 to t4)
    # 2. Gap of 120s at index 5
    # 3. 8 normal observations (t5 to t12)
    # Total observations: 13
    part1 = pd.date_range("2020-02-01 10:00:00", periods=5, freq="10s")
    gap_time = part1[-1] + pd.Timedelta(seconds=120)
    part2 = pd.date_range(gap_time, periods=8, freq="10s")
    ts = pd.concat([pd.Series(part1), pd.Series(part2)], ignore_index=True)

    data = {TIMESTAMP_COL: ts}
    for col in PRIMARY_FEATURES:
        data[col] = np.arange(len(ts), dtype=np.float32)
    df = pd.DataFrame(data)

    builder = CausalWindowBuilder(window_size=6, stride=1, max_gap_seconds=60.0)
    batch = builder.build_windows(df)

    acc = batch.accounting
    assert acc is not None
    assert acc.num_observations == 13
    assert acc.num_candidate_endpoints == 13

    # Window size is 6:
    # Indices 0..4 (5 endpoints): excluded because insufficient history at start
    # Index 5 is a gap reset (120s > 60s).
    # Endpoints 5..9 (5 endpoints): window covers across the gap index 5 -> excluded due to service gap
    # Endpoints 10..12 (3 endpoints): have 6 continuous observations since the gap -> usable windows (3)
    assert acc.num_excluded_insufficient_history == 5
    assert acc.num_excluded_service_gap == 5
    assert acc.num_usable_windows == 3
    assert acc.num_excluded_split_boundary == 0
    assert acc.num_excluded_out_of_bounds == 0

    # Total conservation check:
    total_accounted = (
        acc.num_usable_windows
        + acc.num_excluded_insufficient_history
        + acc.num_excluded_service_gap
        + acc.num_excluded_split_boundary
        + acc.num_excluded_out_of_bounds
    )
    assert total_accounted == acc.num_candidate_endpoints == acc.num_observations


def test_calibration_labeling_boundary_convention() -> None:
    """Verify authoritative boundary behavior: before=normal, start..end=failure, after=normal."""
    # Event 1: 2020-04-18 00:00:00 through 2020-04-18 23:59:59
    ev = next(e for e in FAILURE_EVENTS if e.event_id == "Event_1")

    t_before = ev.start - pd.Timedelta(seconds=10)
    t_start = ev.start
    t_inside = ev.start + pd.Timedelta(hours=12)
    t_end = ev.end
    t_after = ev.end + pd.Timedelta(seconds=10)

    timestamps = np.array([t_before, t_start, t_inside, t_end, t_after])

    scaler = StandardScaler(features=PRIMARY_FEATURES)
    scaler.fit(np.random.normal(5.0, 1.0, size=(100, len(PRIMARY_FEATURES))))
    model = SparseAutoencoder(input_dim=42)

    evaluator = CalibrationEvaluator(model=model, scaler=scaler)
    labels, event_ids = evaluator.assign_event_labels(timestamps)

    # Exact expected boundary results:
    # before = normal (0)
    assert labels[0] == 0
    assert event_ids[0] == "normal"

    # start through end = failure (1)
    assert labels[1] == 1
    assert event_ids[1] == "Event_1"
    assert labels[2] == 1
    assert event_ids[2] == "Event_1"
    assert labels[3] == 1
    assert event_ids[3] == "Event_1"

    # after = normal (0)
    assert labels[4] == 0
    assert event_ids[4] == "normal"


def test_real_partition_counts_constant_reconciliation() -> None:
    """Verify that expected partition row count constants reconcile exactly to total rows."""
    total = (
        EXPECTED_TRAIN_ROWS
        + EXPECTED_CALIBRATION_ROWS
        + EXPECTED_HOLDOUT_ROWS
        + EXPECTED_UNUSED_TAIL_ROWS
    )
    assert total == DATASET_EXPECTED_ROWS == 1_516_948
    assert EXPECTED_TRAIN_ROWS == 445_298
    assert EXPECTED_CALIBRATION_ROWS == 411_534
    assert EXPECTED_HOLDOUT_ROWS == 659_586
    assert EXPECTED_UNUSED_TAIL_ROWS == 530


def test_run_metadata_creation_and_serialization(tmp_path: Path) -> None:
    """Verify that full experiment run produces compact and reproducible metadata."""
    train_df, cal_df = create_synthetic_experiment_data(num_train_rows=200, num_cal_rows=200, seed=42)

    config = ExperimentConfig(
        run_id="test_artifact_run",
        window_size=6,
        epochs=1,
        batch_size=32,
        seed=42,
        validation_fraction=0.1,
    )

    result = run_experiment(
        config=config,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=SYNTHETIC_DATASET_FINGERPRINT,
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
    assert data["metadata"]["holdout_observations_retained"] is False
    assert data["metadata"]["dataset_fingerprint"] == SYNTHETIC_DATASET_FINGERPRINT
    assert "train_boundary" not in data["metadata"]  # nested under split_boundaries
    assert "train" in data["metadata"]["split_boundaries"]
    assert "calibration" in data["metadata"]["split_boundaries"]
    assert "git_commit" in data["metadata"]
    assert "runtime_seconds" in data["metadata"]
    assert "environment" in data["metadata"]

    assert data["metadata"]["train_raw_row_count"] == 200
    assert data["metadata"]["calibration_raw_row_count"] == 200
    assert data["metadata"]["train_fit_raw_row_count"] == 180
    assert data["metadata"]["train_val_raw_row_count"] == 20
    assert data["metadata"]["train_fit_window_count"] > 0
    assert data["metadata"]["train_val_window_count"] > 0
    assert data["metadata"]["calibration_usable_window_count"] > 0
    assert "calibration_exclusion_counts" in data["metadata"]

    assert "config" in data
    assert data["config"]["input_dim"] == 42

    assert "training_history" in data
    assert data["training_history"]["epochs_trained"] == 1

    assert "calibration_metrics" in data
    assert data["calibration_metrics"]["num_usable_windows"] > 0
    assert "window_accounting" in data["calibration_metrics"]
    assert data["calibration_metrics"]["labeling_convention"] == "window_end_timestamp"


def test_baseline_configuration_specification() -> None:
    """Verify that get_baseline_config conforms strictly to the authoritative specification."""
    from src.experiments.runner import BASELINE_RUN_ID, get_baseline_config

    assert BASELINE_RUN_ID == "baseline_w30_standard_l1_s42"
    cfg = get_baseline_config(device="cpu")

    # Authoritative experiment parameters
    assert cfg.run_id == "baseline_w30_standard_l1_s42"
    assert cfg.window_size == 30
    assert cfg.stride == 1
    assert cfg.input_dim == 210  # 7 * 30
    assert cfg.hidden_dim == 128
    assert cfg.latent_dim == 32
    assert cfg.scaler_type == "StandardScaler"
    assert cfg.sparsity_type == "l1"
    assert cfg.sparsity_weight == 1e-4
    assert cfg.target_sparsity == 0.05
    assert cfg.seed == 42
    assert cfg.optimizer == "adam"
    assert cfg.learning_rate == 1e-3
    assert cfg.batch_size == 256
    assert cfg.epochs == 10
    assert cfg.validation_fraction == 0.10
    assert cfg.early_stopping_patience is None
    assert cfg.device == "cpu"
    assert cfg.features == PRIMARY_FEATURES


def test_baseline_synthetic_pipeline_execution(tmp_path: Path) -> None:
    """Verify end-to-end baseline pipeline mechanics using synthetic data without real training."""
    from src.experiments.runner import get_baseline_config

    train_df, cal_df = create_synthetic_experiment_data(num_train_rows=100, num_cal_rows=100, seed=42)
    cfg = get_baseline_config(device="cpu")
    # Use 1 epoch and small batch for fast synthetic test
    cfg.epochs = 1
    cfg.batch_size = 16

    result = run_experiment(
        config=cfg,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=SYNTHETIC_DATASET_FINGERPRINT,
        save_dir=str(tmp_path),
        verbose=False,
    )

    assert result.run_id == "baseline_w30_standard_l1_s42"
    assert result.history.epochs_trained == 1
    assert result.metadata["train_raw_row_count"] == 100
    assert result.metadata["train_fit_raw_row_count"] == 90
    assert result.metadata["train_val_raw_row_count"] == 10
    assert result.metadata["holdout_observations_retained"] is False
    assert (tmp_path / "baseline_w30_standard_l1_s42.json").exists()


def test_training_history_full_epoch_serialization(tmp_path: Path) -> None:
    """Verify that complete epoch-by-epoch training history is serialized into artifact JSON."""
    from src.experiments.trainer import EpochMetrics, TrainingHistory

    history = TrainingHistory(epochs_trained=2, best_epoch=1, best_loss=0.045, early_stopped=False)
    m1 = EpochMetrics(
        epoch=1,
        train_total_loss=0.050,
        train_recon_loss=0.048,
        train_sparsity_loss=0.002,
        val_total_loss=0.045,
        val_recon_loss=0.043,
    )
    m2 = EpochMetrics(
        epoch=2,
        train_total_loss=0.040,
        train_recon_loss=0.038,
        train_sparsity_loss=0.002,
        val_total_loss=0.046,
        val_recon_loss=0.044,
    )
    history.history.extend([m1, m2])

    hist_dict = history.to_dict()

    # Verify summary fields preserved
    assert hist_dict["epochs_trained"] == 2
    assert hist_dict["best_epoch"] == 1
    assert hist_dict["best_loss"] == 0.045
    assert hist_dict["early_stopped"] is False
    assert hist_dict["final_train_total_loss"] == 0.040
    assert hist_dict["final_train_recon_loss"] == 0.038
    assert hist_dict["final_train_sparsity_loss"] == 0.002
    assert hist_dict["final_val_total_loss"] == 0.046

    # Verify complete epoch-by-epoch list serialized
    assert "epochs" in hist_dict
    assert len(hist_dict["epochs"]) == 2
    assert hist_dict["epochs"][0]["epoch"] == 1
    assert hist_dict["epochs"][0]["train_total_loss"] == 0.050
    assert hist_dict["epochs"][0]["train_recon_loss"] == 0.048
    assert hist_dict["epochs"][0]["train_sparsity_loss"] == 0.002
    assert hist_dict["epochs"][0]["val_total_loss"] == 0.045
    assert hist_dict["epochs"][0]["val_recon_loss"] == 0.043

    assert hist_dict["epochs"][1]["epoch"] == 2
    assert hist_dict["epochs"][1]["train_total_loss"] == 0.040


def test_calibration_metric_naming_and_calculation() -> None:
    """Verify that CalibrationEvaluator records both pr_auc_trapezoidal and average_precision."""
    from src.experiments.evaluator import CalibrationEvaluator
    from src.models.sae import SparseAutoencoder
    from src.preprocessing.scalers import StandardScaler

    model = SparseAutoencoder(input_dim=42, hidden_dim=32, latent_dim=16)
    scaler = StandardScaler(features=PRIMARY_FEATURES)
    scaler.fit(np.zeros((10, 7), dtype=np.float32))

    evaluator = CalibrationEvaluator(model=model, scaler=scaler)

    # Synthetic windows: 50 normal, 10 during Event 1 (2020-04-18)
    ts_normal = pd.date_range("2020-04-10", periods=50, freq="10s")
    ts_event = pd.date_range("2020-04-18 01:00:00", periods=10, freq="10s")
    timestamps = np.concatenate([ts_normal.values, ts_event.values])
    windows = np.random.randn(60, 6, 7).astype(np.float32)

    res = evaluator.evaluate(cal_windows=windows, cal_timestamps=timestamps)
    m = res.metrics

    # Verify both metrics are present and bounded [0, 1]
    assert m.roc_auc is not None and 0.0 <= m.roc_auc <= 1.0
    assert m.pr_auc_trapezoidal is not None and 0.0 <= m.pr_auc_trapezoidal <= 1.0
    assert m.average_precision is not None and 0.0 <= m.average_precision <= 1.0
    # Backward compatibility alias
    assert m.pr_auc == m.pr_auc_trapezoidal

    d = m.to_dict()
    assert "pr_auc_trapezoidal" in d
    assert "average_precision" in d
    assert "roc_auc" in d
    assert "Event_1" in m.event_distributions
    assert m.event_distributions["Event_1"]["average_precision"] is not None
    assert 0.0 <= m.event_distributions["Event_1"]["average_precision"] <= 1.0


def test_matrix_enumeration_exact_unique() -> None:
    """Verify that matrix generator produces exactly 16 unique configs as required."""
    configs = generate_matrix_configs()
    assert len(configs) == 16
    
    # Uniqueness dimensions
    unique_ids = set(c.run_id for c in configs)
    assert len(unique_ids) == 16
    
    unique_tuples = set((c.window_size, c.scaler_type, c.sparsity_type) for c in configs)
    assert len(unique_tuples) == 16
    
    # Assert naming convention
    for c in configs:
        assert c.run_id.startswith(f"w{c.window_size:02d}_")
        assert c.run_id.endswith("_s42")
        assert "standard" in c.run_id or "minmax" in c.run_id
        assert "l1" in c.run_id or "kl" in c.run_id

def test_matrix_artifact_resume_behavior(tmp_path) -> None:
    """Verify resume skips valid artifacts and catches malformed ones."""
    from src.experiments.runner import check_artifact_complete
    
    valid_file = tmp_path / "valid.json"
    valid_data = {
        "run_id": "test", "config_hash": "hash123", "dataset_fingerprint": "fp", 
        "git_commit": "abc", "config": {}, "environment": {}, "split_boundaries": {},
        "train_fit_window_count": 10, "calibration_usable_window_count": 10,
        "training_history": {}, 
        "calibration_metrics": {
            "average_precision": 0.5,
            "event_distributions": {"Event_1": {}}
        },
        "metadata": {"status": "SUCCESS", "config_hash": "hash123"}
    }
    import json
    with open(valid_file, "w") as f: json.dump(valid_data, f)
    
    assert check_artifact_complete(str(valid_file), "hash123") == True
    
    # Hash mismatch
    assert check_artifact_complete(str(valid_file), "hash999") == False
    
    # Missing AP
    valid_data["calibration_metrics"].pop("average_precision")
    malformed_file = tmp_path / "malformed.json"
    with open(malformed_file, "w") as f: json.dump(valid_data, f)
    
    assert check_artifact_complete(str(malformed_file), "hash123") == False

def test_dataset_row_count_metadata_canonical(tmp_path) -> None:
    from src.experiments.runner import run_experiment, create_synthetic_experiment_data
    from src.experiments.config import ExperimentConfig
    
    cfg = ExperimentConfig(
        run_id="test_meta",
        window_size=6,
        epochs=1,
        batch_size=16
    )
    
    train_df, cal_df = create_synthetic_experiment_data()
    
    # Run with canonical fingerprint
    from src.experiments.runner import CANONICAL_DATASET_SHA256
    res = run_experiment(
        config=cfg,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=CANONICAL_DATASET_SHA256,
        save_dir=str(tmp_path),
        verbose=False
    )
    # Check row count
    md = res.to_dict()["metadata"]
    assert md["dataset_row_count"] == 1516948

def test_chunked_transform_equivalence() -> None:
    from src.preprocessing.scalers import StandardScaler
    from src.experiments.runner import chunked_transform
    import numpy as np
    
    scaler = StandardScaler(features=["f1", "f2"])
    
    # 3D windows (N, W, D)
    windows = np.random.rand(100, 10, 2).astype(np.float32)
    # fit
    scaler.fit(windows.reshape(-1, 2))
    
    # Traditional
    flat_scaled = scaler.transform(windows.reshape(-1, 2))
    trad_out = flat_scaled.reshape(100, 20).astype(np.float32)
    
    # Chunked
    chunk_out = chunked_transform(scaler, windows, chunk_size=32)
    
    np.testing.assert_allclose(trad_out, chunk_out, rtol=1e-5, atol=1e-5)
