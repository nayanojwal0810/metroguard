"""Reproducible training and calibration experiment runner for MetroGuard Sparse Autoencoders."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.contract import (
    DATASET_SHA256,
    FAILURE_EVENTS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    TIMESTAMP_COL,
    PartitionBoundary,
    SplitPartition,
)
from src.data.validator import assign_split, validate_dataset
from src.data.windowing import CausalWindowBuilder
from src.experiments.config import ExperimentConfig, generate_matrix_configs
from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, CalibrationResult
from src.experiments.trainer import SAETrainer, TrainingHistory, set_seed
from src.models.sae import SparseAutoencoder
from src.preprocessing.scalers import BaseScaler, MinMaxScaler, StandardScaler


@dataclass
class ExperimentResult:
    """Consolidated results, metadata, and artifact representation for an experiment run."""

    run_id: str
    config: ExperimentConfig
    history: TrainingHistory
    calibration_metrics: CalibrationMetrics
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire experiment result to serializable JSON-friendly dictionary."""
        return {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "config": self.config.to_dict(),
            "training_history": self.history.to_dict(),
            "calibration_metrics": self.calibration_metrics.to_dict(),
        }

    def save(self, filepath: str) -> None:
        """Save compact experiment result metadata to JSON file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_git_commit() -> str:
    """Retrieve current Git HEAD commit hash."""
    try:
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent.parent
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
        return res.stdout.strip()
    except Exception:
        return "unversioned"


def get_environment_info() -> Dict[str, str]:
    """Capture runtime environment and library versions."""
    import platform
    import sklearn

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }


def create_synthetic_experiment_data(
    num_train_rows: int = 500,
    num_cal_rows: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic TRAIN and CALIBRATION DataFrames for fast testing.

    Includes:
    - Train period in Feb 2020.
    - Calibration period in Apr 2020, with timestamps covering Event 1.
    - Valid timestamps and monotonic cadence (10s).
    - All 7 primary features.
    """
    rng = np.random.RandomState(seed)

    # 1. TRAIN dataframe (Feb 2020)
    train_ts = pd.date_range(
        start="2020-02-01 08:00:00",
        periods=num_train_rows,
        freq="10s",
    )
    train_dict: Dict[str, Any] = {TIMESTAMP_COL: train_ts}
    for col in PRIMARY_FEATURES:
        train_dict[col] = rng.normal(loc=5.0, scale=1.0, size=num_train_rows).astype(
            np.float32
        )
    train_df = pd.DataFrame(train_dict)

    # 2. CALIBRATION dataframe (starts Apr 17 to span both normal and Event 1)
    cal_ts = pd.date_range(
        start="2020-04-17 23:40:00",
        periods=num_cal_rows,
        freq="10s",
    )
    is_event = cal_ts >= pd.Timestamp("2020-04-18 00:00:00")
    cal_dict: Dict[str, Any] = {TIMESTAMP_COL: cal_ts}
    for col in PRIMARY_FEATURES:
        base = rng.normal(loc=5.0, scale=1.0, size=num_cal_rows).astype(np.float32)
        base[is_event] += 4.0  # Anomaly shift during failure event
        cal_dict[col] = base
    cal_df = pd.DataFrame(cal_dict)

    return train_df, cal_df


def load_train_and_calibration_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw dataset and return strictly TRAIN and CALIBRATION partitions.

    CRITICAL LEAKAGE RULE:
    FINAL HOLDOUT data (>= 2020-06-01) is strictly filtered out and never loaded
    into the experiment pipeline.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw CSV file not found at: {path}")

    # Read raw data
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    # Assign split partitions
    df["partition"] = assign_split(df[TIMESTAMP_COL])

    # Enforce holdout protection: Drop holdout and unused tail entirely
    train_df = df[df["partition"] == SplitPartition.TRAIN.value].copy()
    cal_df = df[df["partition"] == SplitPartition.CALIBRATION.value].copy()

    # Drop partition column to maintain clean feature schemas
    train_df = train_df.drop(columns=["partition"])
    cal_df = cal_df.drop(columns=["partition"])

    # Strict assertion verifying holdout exclusion
    train_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.TRAIN)
    cal_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.CALIBRATION)

    assert (train_df[TIMESTAMP_COL] <= train_bound.end).all(), "Train partition leak detected!"
    assert (cal_df[TIMESTAMP_COL] <= cal_bound.end).all(), "Calibration partition leak detected!"
    assert (cal_df[TIMESTAMP_COL] >= cal_bound.start).all(), "Calibration start violation!"

    return train_df, cal_df


def run_experiment(
    config: ExperimentConfig,
    train_df: Optional[pd.DataFrame] = None,
    cal_df: Optional[pd.DataFrame] = None,
    raw_csv_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
) -> ExperimentResult:
    """Execute end-to-end reproducible experiment for a single configuration.

    Workflow:
    1. Prepare train & cal data (synthetic or filtered from raw CSV).
    2. Build causal rolling windows (strictly respecting gaps and boundaries).
    3. Fit scaler strictly on TRAIN data.
    4. Transform TRAIN data and train SAE model with SAETrainer.
    5. Evaluate CALIBRATION data using CalibrationEvaluator (scaler & model frozen).
    6. Compute metrics and serialize run artifact.
    """
    start_time = time.time()
    set_seed(config.seed)

    # 1. Obtain data
    dataset_fingerprint = "synthetic"
    if train_df is None or cal_df is None:
        if raw_csv_path is None:
            raw_csv_path = "data/raw/MetroPT3(AirCompressor).csv"
        train_df, cal_df = load_train_and_calibration_data(raw_csv_path)
        dataset_fingerprint = DATASET_SHA256

    if verbose:
        print(f"[{config.run_id}] Train rows: {len(train_df)}, Cal rows: {len(cal_df)}")

    # 2. Window construction
    builder = CausalWindowBuilder(
        window_size=config.window_size,
        stride=config.stride,
        features=config.features,
    )
    train_batch = builder.build_windows(train_df)
    cal_batch = builder.build_windows(cal_df)

    if len(train_batch.windows) == 0:
        raise ValueError(
            f"No usable train windows produced for window_size={config.window_size}"
        )
    if len(cal_batch.windows) == 0:
        raise ValueError(
            f"No usable calibration windows produced for window_size={config.window_size}"
        )

    if verbose:
        print(
            f"[{config.run_id}] Built {len(train_batch.windows)} train windows, "
            f"{len(cal_batch.windows)} cal windows"
        )

    # 3. Scaler fitting strictly on TRAIN windows
    n_train_w, w_len, n_feat = train_batch.windows.shape
    train_flat_features = train_batch.windows.reshape(-1, n_feat)

    if config.scaler_type == "StandardScaler":
        scaler: BaseScaler = StandardScaler(features=config.features)
    elif config.scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler(features=config.features)
    else:
        raise ValueError(f"Unknown scaler_type: {config.scaler_type}")

    scaler.fit(train_flat_features)

    # Transform TRAIN windows and reshape to (N, W * D)
    scaled_train_flat = scaler.transform(train_flat_features)
    scaled_train_windows = scaled_train_flat.reshape(n_train_w, w_len * n_feat)

    # 4. Prepare PyTorch DataLoaders (Train / Train-derived Validation)
    train_tensor = torch.from_numpy(scaled_train_windows).float()

    if config.validation_fraction > 0.0:
        val_size = int(len(train_tensor) * config.validation_fraction)
        val_size = max(1, val_size)
        # Chronological holdout from train
        tr_data = train_tensor[:-val_size]
        val_data = train_tensor[-val_size:]
        train_loader = DataLoader(
            TensorDataset(tr_data),
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=config.batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = None

    # 5. Initialize model and train
    model = SparseAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        sparsity_type=config.sparsity_type,
        sparsity_weight=config.sparsity_weight,
        target_sparsity=config.target_sparsity,
    )

    trainer = SAETrainer(
        model=model,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer,
        device=config.device,
        seed=config.seed,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        early_stopping_patience=config.early_stopping_patience,
    )

    # 6. Evaluate on CALIBRATION (model & scaler frozen)
    evaluator = CalibrationEvaluator(
        model=model,
        scaler=scaler,
        device=config.device,
    )
    cal_result: CalibrationResult = evaluator.evaluate(
        cal_windows=cal_batch.windows,
        cal_timestamps=cal_batch.timestamps,
        num_calibration_rows=len(cal_df),
        batch_size=config.batch_size,
    )

    elapsed_seconds = round(time.time() - start_time, 2)

    # 7. Compile run metadata
    train_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.TRAIN)
    cal_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.CALIBRATION)

    metadata: Dict[str, Any] = {
        "run_id": config.run_id,
        "config_hash": config.compute_config_hash(),
        "git_commit": get_git_commit(),
        "seed": config.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "train_boundary": {"start": str(train_bound.start), "end": str(train_bound.end)},
        "calibration_boundary": {"start": str(cal_bound.start), "end": str(cal_bound.end)},
        "num_train_windows": len(train_batch.windows),
        "num_calibration_windows": len(cal_batch.windows),
        "runtime_seconds": elapsed_seconds,
        "environment": get_environment_info(),
        "holdout_accessed": False,
    }

    result = ExperimentResult(
        run_id=config.run_id,
        config=config,
        history=history,
        calibration_metrics=cal_result.metrics,
        metadata=metadata,
    )

    # 8. Save artifact if directory is provided
    if save_dir is not None:
        out_file = Path(save_dir) / f"{config.run_id}.json"
        result.save(str(out_file))
        if verbose:
            print(f"[{config.run_id}] Saved metadata artifact to: {out_file}")

    return result


def run_matrix(
    configs: Sequence[ExperimentConfig],
    raw_csv_path: Optional[str] = None,
    save_dir: str = "artifacts/experiments",
    verbose: bool = True,
) -> List[ExperimentResult]:
    """Execute multiple configurations sequentially."""
    results: List[ExperimentResult] = []
    # Load data once if loading from CSV to avoid redundant parsing
    train_df: Optional[pd.DataFrame] = None
    cal_df: Optional[pd.DataFrame] = None

    if raw_csv_path is not None:
        train_df, cal_df = load_train_and_calibration_data(raw_csv_path)

    for i, cfg in enumerate(configs, start=1):
        if verbose:
            print(f"\n--- Running experiment {i}/{len(configs)}: {cfg.run_id} ---")
        res = run_experiment(
            config=cfg,
            train_df=train_df,
            cal_df=cal_df,
            raw_csv_path=raw_csv_path,
            save_dir=save_dir,
            verbose=verbose,
        )
        results.append(res)
    return results


def run_smoke_test(verbose: bool = True) -> ExperimentResult:
    """Execute a lightweight synthetic smoke test verifying the end-to-end pipeline.

    Takes < 2 seconds on CPU.
    """
    if verbose:
        print("=== MetroGuard Experiment Pipeline: Smoke Test ===")
        print("Generating synthetic TRAIN and CALIBRATION datasets...")

    train_df, cal_df = create_synthetic_experiment_data(
        num_train_rows=300,
        num_cal_rows=300,
        seed=42,
    )

    config = ExperimentConfig(
        run_id="smoke_test_run",
        window_size=6,
        stride=1,
        scaler_type="StandardScaler",
        sparsity_type="l1",
        sparsity_weight=1e-4,
        epochs=2,
        batch_size=32,
        seed=42,
    )

    if verbose:
        print(f"Executing configuration: {config.run_id} (D={config.input_dim}, H={config.hidden_dim}, Z={config.latent_dim})")

    result = run_experiment(
        config=config,
        train_df=train_df,
        cal_df=cal_df,
        save_dir=None,
        verbose=verbose,
    )

    if verbose:
        m = result.calibration_metrics
        print("\n--- Smoke Test Succeeded ---")
        print(f"Epochs trained: {result.history.epochs_trained}")
        print(f"Final train total loss: {result.history.history[-1].train_total_loss:.6f}")
        print(f"Final train recon loss: {result.history.history[-1].train_recon_loss:.6f}")
        print(f"Usable cal windows: {m.num_usable_windows}")
        print(f"Failure cal windows: {m.num_failure_windows}")
        print(f"Normal mean recon error: {m.normal_distribution.get('mean', 0.0):.6f}")
        print(f"Failure mean recon error: {m.failure_distribution.get('mean', 0.0):.6f}")
        if m.pr_auc is not None:
            print(f"PR-AUC: {m.pr_auc:.4f}")
        if m.roc_auc is not None:
            print(f"ROC-AUC: {m.roc_auc:.4f}")
        print(f"Runtime: {result.metadata['runtime_seconds']}s")
        print("Holdout accessed: FALSE")

    return result


def main() -> None:
    """CLI entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="MetroGuard SAE Baseline Training & Calibration Experiment Runner"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight end-to-end smoke test on synthetic data.",
    )
    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="List all 16 candidate baseline configurations in the experiment matrix.",
    )
    parser.add_argument(
        "--estimate-resources",
        action="store_true",
        help="Display estimated compute and memory requirements for real-data experiments.",
    )

    args = parser.parse_args()

    if args.smoke:
        run_smoke_test(verbose=True)
    elif args.list_matrix:
        configs = generate_matrix_configs()
        print(f"Total candidate matrix configurations: {len(configs)}")
        for idx, cfg in enumerate(configs, 1):
            print(
                f"{idx:2d}. {cfg.run_id:<28} | W={cfg.window_size:<3} | D={cfg.input_dim:<4} | "
                f"H={cfg.hidden_dim:<3} | Z={cfg.latent_dim:<3} | Scaler={cfg.scaler_type:<14} | "
                f"Sparsity={cfg.sparsity_type:<2} (wt={cfg.sparsity_weight})"
            )
    elif args.estimate_resources:
        print("=== Estimated Resource Requirements: Real-Data Experiments ===")
        print("Dataset: MetroPT3 (445k TRAIN rows, 446k CALIBRATION rows)")
        print("Hardware Target: Single CPU Core (x86_64)")
        print("- Single Run (10 epochs, batch size 256): ~3 to 5 minutes, ~1.5 GB RAM")
        print("- 16-Config Matrix Sequential: ~48 to 80 minutes, ~2.5 GB peak RAM")
        print("Hardware Target: Google Colab / GPU (T4 / V100)")
        print("- Single Run: ~20 to 30 seconds")
        print("- 16-Config Matrix: ~5 to 8 minutes")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
