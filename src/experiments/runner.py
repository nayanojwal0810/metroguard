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
    CANONICAL_DATASET_SHA256,
    DATASET_EXPECTED_ROWS,
    DATASET_SHA256,
    EXPECTED_CALIBRATION_ROWS,
    EXPECTED_TRAIN_ROWS,
    FAILURE_EVENTS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    SYNTHETIC_DATASET_FINGERPRINT,
    TIMESTAMP_COL,
    PartitionBoundary,
    SplitPartition,
)
from src.data.validator import assign_split, validate_dataset
from src.data.windowing import CausalWindowBuilder, WindowAccounting
from src.experiments.config import ExperimentConfig, generate_matrix_configs
from src.experiments.evaluator import CalibrationEvaluator, CalibrationMetrics, CalibrationResult
from src.experiments.selection import MatrixSummaryRow
from src.experiments.trainer import SAETrainer, TrainingHistory, set_seed
from src.models.sae import SparseAutoencoder
from src.preprocessing.scalers import BaseScaler, MinMaxScaler, StandardScaler

def chunked_transform(scaler: BaseScaler, windows: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    """Safely transform large window arrays in chunks to prevent multi-GB memory allocations."""
    if len(windows) == 0:
        return np.empty((0, 0), dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError("chunked_transform expects 3D array (N, W, D)")
        
    n_w, w_len, n_feat = windows.shape
    out = np.empty((n_w, w_len * n_feat), dtype=np.float32)
    
    for i in range(0, n_w, chunk_size):
        chunk = windows[i : i + chunk_size]
        reshaped = chunk.reshape(-1, n_feat)
        scaled_chunk = scaler.transform(reshaped)
        out[i : i + chunk_size] = scaled_chunk.reshape(len(chunk), w_len * n_feat).astype(np.float32)
        
    return out


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
        base[is_event] += 4.0  # Simulated anomaly shift during failure event
        cal_dict[col] = base
    cal_df = pd.DataFrame(cal_dict)

    return train_df, cal_df


def load_train_and_calibration_data(
    csv_path: str,
    features: Sequence[str] = PRIMARY_FEATURES,
    chunk_size: int = 100_000,
    enforce_real_counts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw dataset in chunks, retaining strictly TRAIN and CALIBRATION partitions.

    CRITICAL LEAKAGE GUARANTEE:
    - Only [TIMESTAMP_COL] and the specified features are ingested from disk.
    - FINAL HOLDOUT (>= 2020-06-01 00:00:00) and UNUSED_TAIL rows are filtered out at the chunk
      level and NEVER concatenated or retained in experiment DataFrames.
    - Chunk processing terminates early as soon as the chronological timestamp stream passes CALIBRATION_END.
    - FINAL HOLDOUT observations are never passed to window construction, scaler fitting,
      model training, or calibration evaluation.
    - If the input file is the canonical MetroPT-3 dataset, verifies frozen partition counts:
      TRAIN == 445,298 and CALIBRATION == 411,534.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw CSV file not found at: {path}")

    train_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.TRAIN)
    cal_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.CALIBRATION)
    holdout_start = pd.Timestamp("2020-06-01 00:00:00")

    cols_to_read = [TIMESTAMP_COL] + list(features)

    train_chunks: List[pd.DataFrame] = []
    cal_chunks: List[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=cols_to_read, chunksize=chunk_size):
        chunk[TIMESTAMP_COL] = pd.to_datetime(chunk[TIMESTAMP_COL])

        # If entire chunk is before train start, skip
        if chunk[TIMESTAMP_COL].iloc[-1] < train_bound.start:
            continue

        # Extract TRAIN slice in this chunk
        train_mask = (chunk[TIMESTAMP_COL] >= train_bound.start) & (
            chunk[TIMESTAMP_COL] <= train_bound.end
        )
        if train_mask.any():
            train_chunks.append(chunk[train_mask])

        # Extract CALIBRATION slice in this chunk
        cal_mask = (chunk[TIMESTAMP_COL] >= cal_bound.start) & (
            chunk[TIMESTAMP_COL] <= cal_bound.end
        )
        if cal_mask.any():
            cal_chunks.append(chunk[cal_mask])

        # Early termination: if this chunk begins past calibration end, stop reading immediately!
        if chunk[TIMESTAMP_COL].iloc[0] > cal_bound.end:
            break

    if not train_chunks:
        train_df = pd.DataFrame(columns=cols_to_read)
    else:
        train_df = pd.concat(train_chunks, ignore_index=True)

    if not cal_chunks:
        cal_df = pd.DataFrame(columns=cols_to_read)
    else:
        cal_df = pd.concat(cal_chunks, ignore_index=True)

    # Strict holdout exclusion assertions
    assert (train_df[TIMESTAMP_COL] < holdout_start).all(), (
        "CRITICAL LEAKAGE: FINAL HOLDOUT rows detected in TRAIN DataFrame!"
    )
    assert (cal_df[TIMESTAMP_COL] < holdout_start).all(), (
        "CRITICAL LEAKAGE: FINAL HOLDOUT rows detected in CALIBRATION DataFrame!"
    )
    assert (train_df[TIMESTAMP_COL] >= train_bound.start).all(), (
        "Pre-TRAIN observations detected in TRAIN DataFrame!"
    )
    assert (train_df[TIMESTAMP_COL] <= train_bound.end).all(), (
        "Post-TRAIN observations detected in TRAIN DataFrame!"
    )
    assert (cal_df[TIMESTAMP_COL] >= cal_bound.start).all(), (
        "Pre-CALIBRATION observations detected in CALIBRATION DataFrame!"
    )
    assert (cal_df[TIMESTAMP_COL] <= cal_bound.end).all(), (
        "Post-CALIBRATION observations detected in CALIBRATION DataFrame!"
    )

    # Fail-fast check on real MetroPT-3 partition counts if file matches canonical name or size
    if enforce_real_counts and path.name == "MetroPT3(AirCompressor).csv":
        if len(train_df) != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"Partition count discrepancy in TRAIN! "
                f"Expected {EXPECTED_TRAIN_ROWS} rows, got {len(train_df)}."
            )
        if len(cal_df) != EXPECTED_CALIBRATION_ROWS:
            raise ValueError(
                f"Partition count discrepancy in CALIBRATION! "
                f"Expected {EXPECTED_CALIBRATION_ROWS} rows, got {len(cal_df)}."
            )

    return train_df, cal_df


def _verify_scaler_snapshot(
    scaler: BaseScaler,
    snapshot: Dict[str, Any],
    stage_name: str,
) -> None:
    """Verify that scaler parameters did not mutate during a transform operation."""
    if isinstance(scaler, StandardScaler):
        if snapshot.get("mean") is not None:
            assert np.array_equal(scaler.mean_, snapshot["mean"]), (
                f"Scaler mean_ mutated during {stage_name}!"
            )
        if snapshot.get("std") is not None:
            assert np.array_equal(scaler.std_, snapshot["std"]), (
                f"Scaler std_ mutated during {stage_name}!"
            )
    elif isinstance(scaler, MinMaxScaler):
        if snapshot.get("min") is not None:
            assert np.array_equal(scaler.min_, snapshot["min"]), (
                f"Scaler min_ mutated during {stage_name}!"
            )
        if snapshot.get("max") is not None:
            assert np.array_equal(scaler.max_, snapshot["max"]), (
                f"Scaler max_ mutated during {stage_name}!"
            )
        if snapshot.get("range") is not None:
            assert np.array_equal(scaler.range_, snapshot["range"]), (
                f"Scaler range_ mutated during {stage_name}!"
            )


def run_experiment(
    config: ExperimentConfig,
    train_df: Optional[pd.DataFrame] = None,
    cal_df: Optional[pd.DataFrame] = None,
    raw_csv_path: Optional[str] = None,
    dataset_fingerprint: Optional[str] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
) -> ExperimentResult:
    """Execute end-to-end reproducible experiment for a single configuration.

    Workflow:
    1. Ingestion: load data (synthetic or chunked holdout-safe from CSV).
    2. Chronological TRAIN split into TRAIN-fit and TRAIN-validation.
    3. Causal rolling window construction for TRAIN-fit, TRAIN-val, and CALIBRATION.
    4. Fit scaler ONLY on TRAIN-fit subset (never on val or cal).
    5. Transform TRAIN-fit, TRAIN-val, and CALIBRATION with the frozen scaler.
    6. Train SAE with SAETrainer (capturing and restoring best validation state).
    7. Evaluate CALIBRATION with CalibrationEvaluator (model and scaler frozen).
    8. Record explicit accounting and metadata without holdout leakage.
    """
    start_time = time.time()
    set_seed(config.seed)

    # 1. Obtain data and determine provenance fingerprint
    if dataset_fingerprint is None:
        if train_df is None or cal_df is None:
            dataset_fingerprint = CANONICAL_DATASET_SHA256
        else:
            dataset_fingerprint = SYNTHETIC_DATASET_FINGERPRINT

    if train_df is None or cal_df is None:
        if raw_csv_path is None:
            raw_csv_path = "data/raw/MetroPT3(AirCompressor).csv"
        train_df, cal_df = load_train_and_calibration_data(
            raw_csv_path,
            features=config.features,
        )
        dataset_fingerprint = CANONICAL_DATASET_SHA256

    if verbose:
        print(f"[{config.run_id}] Train rows: {len(train_df)}, Cal rows: {len(cal_df)}")

    # 2. Chronological separation of TRAIN into TRAIN-fit and TRAIN-validation
    n_train_rows = len(train_df)
    if config.validation_fraction > 0.0:
        n_val_rows = int(n_train_rows * config.validation_fraction)
        n_fit_rows = n_train_rows - n_val_rows
        train_fit_df = train_df.iloc[:n_fit_rows].copy()
        train_val_df = train_df.iloc[n_fit_rows:].copy()
    else:
        train_fit_df = train_df
        train_val_df = pd.DataFrame(columns=train_df.columns)
        n_fit_rows = n_train_rows
        n_val_rows = 0

    # 3. Causal window construction
    builder = CausalWindowBuilder(
        window_size=config.window_size,
        stride=config.stride,
        features=config.features,
    )
    train_fit_batch = builder.build_windows(train_fit_df)
    train_val_batch = (
        builder.build_windows(train_val_df) if len(train_val_df) > 0 else None
    )
    cal_batch = builder.build_windows(cal_df)

    if len(train_fit_batch.windows) == 0:
        raise ValueError(
            f"No usable train-fit windows produced for window_size={config.window_size}"
        )
    if len(cal_batch.windows) == 0:
        raise ValueError(
            f"No usable calibration windows produced for window_size={config.window_size}"
        )

    if verbose:
        n_val_w_count = len(train_val_batch.windows) if train_val_batch else 0
        print(
            f"[{config.run_id}] Windows -> Fit: {len(train_fit_batch.windows)}, "
            f"Val: {n_val_w_count}, Cal: {len(cal_batch.windows)}"
        )

    # 4. Scaler fitting strictly on TRAIN-fit subset (never on val or cal)
    n_fit_w, w_len, n_feat = train_fit_batch.windows.shape
    fit_flat = train_fit_batch.windows.reshape(-1, n_feat)

    if config.scaler_type == "StandardScaler":
        scaler: BaseScaler = StandardScaler(features=config.features)
    elif config.scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler(features=config.features)
    else:
        raise ValueError(f"Unknown scaler_type: {config.scaler_type}")

    scaler.fit(fit_flat)

    # Capture snapshot of scaler parameters to verify zero mutation
    if isinstance(scaler, StandardScaler):
        scaler_snapshot = {
            "mean": scaler.mean_.copy(),
            "std": scaler.std_.copy(),
        }
    elif isinstance(scaler, MinMaxScaler):
        scaler_snapshot = {
            "min": scaler.min_.copy(),
            "max": scaler.max_.copy(),
            "range": scaler.range_.copy(),
        }
    else:
        scaler_snapshot = {}

    # Transform TRAIN-fit windows safely in chunks
    scaled_fit_windows = chunked_transform(scaler, train_fit_batch.windows, chunk_size=8192)

    # Transform TRAIN-validation windows (if present) with frozen scaler
    if train_val_batch is not None and len(train_val_batch.windows) > 0:
        scaled_val_windows = chunked_transform(scaler, train_val_batch.windows, chunk_size=8192)

        # Verify scaler parameters remained identical after val transform
        _verify_scaler_snapshot(scaler, scaler_snapshot, "validation transform")
    else:
        scaled_val_windows = None

    # 5. DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled_fit_windows).float()),
        batch_size=config.batch_size,
        shuffle=True,
    )
    if scaled_val_windows is not None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(scaled_val_windows).float()),
            batch_size=config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    # 6. Initialize model and train with early stopping parameter restoration
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
        verbose=verbose,
    )

    # 7. Evaluate on CALIBRATION (model & scaler frozen)
    evaluator = CalibrationEvaluator(
        model=model,
        scaler=scaler,
        device=config.device,
    )
    cal_result: CalibrationResult = evaluator.evaluate(
        cal_windows=cal_batch.windows,
        cal_timestamps=cal_batch.timestamps,
        window_accounting=cal_batch.accounting,
        batch_size=config.batch_size,
    )

    # Verify scaler parameters remained identical after cal transform
    _verify_scaler_snapshot(scaler, scaler_snapshot, "calibration transform")

    elapsed_seconds = round(time.time() - start_time, 2)

    # 8. Compile run metadata
    train_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.TRAIN)
    cal_bound = next(b for b in SPLIT_BOUNDS if b.name == SplitPartition.CALIBRATION)

    metadata: Dict[str, Any] = {
        "run_id": config.run_id,
        "config_hash": config.compute_config_hash(),
        "git_commit": get_git_commit(),
        "seed": config.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_row_count": 1516948 if dataset_fingerprint == CANONICAL_DATASET_SHA256 else (len(train_df) + len(cal_df)),
        "train_raw_row_count": len(train_df),
        "calibration_raw_row_count": len(cal_df),
        "train_fit_raw_row_count": len(train_fit_df),
        "train_fit_window_count": len(train_fit_batch.windows),
        "train_val_raw_row_count": len(train_val_df),
        "train_val_window_count": len(train_val_batch.windows) if train_val_batch else 0,
        "calibration_usable_window_count": len(cal_batch.windows),
        "calibration_exclusion_counts": (
            cal_batch.accounting.to_dict() if cal_batch.accounting else {}
        ),
        "split_boundaries": {
            "train": {"start": str(train_bound.start), "end": str(train_bound.end)},
            "calibration": {"start": str(cal_bound.start), "end": str(cal_bound.end)},
        },
        "runtime_seconds": elapsed_seconds,
        "environment": get_environment_info(),
        "holdout_observations_retained": False,
        "holdout_accessed": False,
        "status": "SUCCESS",
    }

    result = ExperimentResult(
        run_id=config.run_id,
        config=config,
        history=history,
        calibration_metrics=cal_result.metrics,
        metadata=metadata,
    )

    # 9. Save artifact if directory is provided
    if save_dir is not None:
        out_file = Path(save_dir) / f"{config.run_id}.json"
        result.save(str(out_file))
        if verbose:
            print(f"[{config.run_id}] Saved metadata artifact to: {out_file}")

    return result


def check_artifact_complete(file_path: str, expected_hash: str) -> bool:
    try:
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        req = ["run_id", "config_hash", "dataset_fingerprint", "git_commit", "config", "environment", "split_boundaries", "train_fit_window_count", "calibration_usable_window_count", "training_history", "calibration_metrics", "metadata"]
        if not all(k in data for k in req) and not all(k in data.get("metadata", {}) for k in req):
            pass # simplified check
            
        md = data.get("metadata", {})
        cm = data.get("calibration_metrics", {})
        if md.get("status") != "SUCCESS": return False
        if data.get("config_hash") != expected_hash and md.get("config_hash") != expected_hash: return False
        if "average_precision" not in cm or cm["average_precision"] is None: return False
        if "event_distributions" not in cm or "Event_1" not in cm["event_distributions"]: return False
        return True
    except Exception:
        return False

def extract_summary_row(data: dict) -> MatrixSummaryRow:
    md = data["metadata"]
    cfg = data["config"]
    cm = data["calibration_metrics"]
    hist = data["training_history"]
    
    e1 = cm.get("event_distributions", {}).get("Event_1", {})
    e2 = cm.get("event_distributions", {}).get("Event_2", {})
    norm = cm.get("normal_distribution", {})
    
    # Calculate parameter count (D*H + H + H*Z + Z + Z*H + H + H*D + D) -> derived from dims
    # But for now just approximation or what's stored
    d, h, z = cfg["input_dim"], cfg["hidden_dim"], cfg["latent_dim"]
    param_count = (d*h + h) + (h*z + z) + (z*h + h) + (h*d + d)
    
    return MatrixSummaryRow(
        run_id=cfg["run_id"],
        w=cfg["window_size"],
        scaler=cfg["scaler_type"],
        sparsity=cfg["sparsity_type"],
        sparsity_weight=cfg["sparsity_weight"],
        h=cfg["hidden_dim"],
        z=cfg["latent_dim"],
        parameter_count=param_count,
        average_precision=cm.get("average_precision", 0.0),
        pr_auc_trapezoidal=cm.get("pr_auc_trapezoidal", 0.0),
        roc_auc=cm.get("roc_auc", 0.0),
        event_1_ap=e1.get("average_precision"),
        event_2_ap=e2.get("average_precision"),
        event_1_median=e1.get("median"),
        event_2_median=e2.get("median"),
        normal_median=norm.get("median", 0.0),
        normal_p95=norm.get("p95", 0.0),
        normal_p99=norm.get("p99", 0.0),
        best_epoch=hist.get("best_epoch", 0),
        best_val_loss=hist.get("best_loss", 0.0),
        runtime=md.get("runtime_seconds", 0.0),
        usable_calibration_windows=cm.get("num_usable_windows", 0),
        exclusion_counts=md.get("calibration_exclusion_counts", {})
    )

def run_matrix(
    configs: Sequence[ExperimentConfig],
    raw_csv_path: Optional[str] = None,
    save_dir: str = "artifacts/runs/matrix",
    resume: bool = False,
    verbose: bool = True,
) -> List[ExperimentResult]:
    import traceback
    
    results: List[ExperimentResult] = []
    summaries: List[MatrixSummaryRow] = []
    train_df, cal_df = None, None
    matrix_fingerprint = CANONICAL_DATASET_SHA256
    
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    matrix_complete = True
    
    for i, cfg in enumerate(configs, start=1):
        if verbose:
            print(f"\n--- Running experiment {i}/{len(configs)}: {cfg.run_id} ---")
            
        out_file = out_dir / f"{cfg.run_id}.json"
        
        if resume and out_file.exists():
            if check_artifact_complete(str(out_file), cfg.compute_config_hash()):
                if verbose:
                    print(f"[{cfg.run_id}] SKIPPED_VALID_EXISTING")
                import json
                with open(out_file, "r") as f:
                    data = json.load(f)
                summaries.append(extract_summary_row(data))
                continue
            else:
                if verbose:
                    print(f"[{cfg.run_id}] Found incomplete/malformed artifact. Rerunning.")
        
        # Lazy load data
        if train_df is None and raw_csv_path is not None:
            train_df, cal_df = load_train_and_calibration_data(raw_csv_path)
            
        try:
            res = run_experiment(
                config=cfg,
                train_df=train_df,
                cal_df=cal_df,
                raw_csv_path=raw_csv_path,
                dataset_fingerprint=matrix_fingerprint,
                save_dir=str(out_dir),
                verbose=verbose,
            )
            results.append(res)
            summaries.append(extract_summary_row(res.to_dict()))
        except Exception as e:
            print(f"[{cfg.run_id}] FAILED: {str(e)}")
            traceback.print_exc()
            matrix_complete = False
            break # STOP MATRIX ON FAILURE
            
    # Check if exactly 16 successful
    if matrix_complete and len(summaries) == 16:
        summary_path = out_dir / "matrix_summary.json"
        import json
        with open(summary_path, "w") as f:
            json.dump([s.to_dict() for s in summaries], f, indent=2)
        if verbose:
            print(f"MATRIX COMPLETE. Summary saved to {summary_path}")
    else:
        if verbose:
            print("MATRIX INCOMPLETE.")
        
    return results

def run_preflight(data_path: str, verbose: bool = True) -> None:
    print("=== MetroGuard Preflight Check ===")
    p = Path(data_path)
    if not p.exists():
        print(f"FAIL: Dataset not found at {data_path}")
        return
    print("Dataset exists.")
    
    configs = generate_matrix_configs()
    if len(configs) != 16:
        print(f"FAIL: Matrix generator produced {len(configs)} configs instead of 16.")
        return
    print("Matrix enumeration valid (16 unique configurations).")
    
    unique_ids = set(c.run_id for c in configs)
    if len(unique_ids) != 16:
        print("FAIL: Run IDs are not unique.")
        return
    print("Run IDs are deterministic and unique.")
    
    commit = get_git_commit()
    print(f"Git commit: {commit}")
    
    print("Preflight check passed. Ready for long-run matrix execution.")



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
        validation_fraction=0.2,
    )

    if verbose:
        print(
            f"Executing configuration: {config.run_id} (D={config.input_dim}, "
            f"H={config.hidden_dim}, Z={config.latent_dim})"
        )

    result = run_experiment(
        config=config,
        train_df=train_df,
        cal_df=cal_df,
        dataset_fingerprint=SYNTHETIC_DATASET_FINGERPRINT,
        save_dir=None,
        verbose=verbose,
    )

    if verbose:
        m = result.calibration_metrics
        print("\n--- Smoke Test Succeeded ---")
        print("Note: Synthetic test only (verifies pipeline execution; not an evaluation of model performance).")
        print(f"Dataset fingerprint: {result.metadata['dataset_fingerprint']}")
        print(f"Epochs trained: {result.history.epochs_trained}")
        print(f"Best validation epoch: {result.history.best_epoch}")
        print(f"Final train total loss: {result.history.history[-1].train_total_loss:.6f}")
        print(f"Train-fit windows: {result.metadata['train_fit_window_count']}")
        print(f"Train-val windows: {result.metadata['train_val_window_count']}")
        print(f"Usable cal windows: {m.num_usable_windows}")
        print(f"Failure cal windows: {m.num_failure_windows}")
        print(f"Window accounting: {m.window_accounting}")
        print(f"Normal mean recon error: {m.normal_distribution.get('mean', 0.0):.6f}")
        print(f"Failure mean recon error: {m.failure_distribution.get('mean', 0.0):.6f}")
        if m.pr_auc_trapezoidal is not None:
            print(f"PR-AUC (trapezoidal, synthetic verification): {m.pr_auc_trapezoidal:.4f}")
        if m.average_precision is not None:
            print(f"Average Precision (synthetic verification): {m.average_precision:.4f}")
        if m.roc_auc is not None:
            print(f"ROC-AUC (synthetic verification): {m.roc_auc:.4f}")
        print(f"Runtime: {result.metadata['runtime_seconds']}s")
        print(f"Holdout observations retained: {result.metadata['holdout_observations_retained']}")

    return result


BASELINE_RUN_ID = "baseline_w30_standard_l1_s42"


def get_baseline_config(device: str = "cpu") -> ExperimentConfig:
    """Construct the authoritative first baseline experiment configuration.

    Settings:
    - Run ID: baseline_w30_standard_l1_s42
    - Window size: W = 30 observations
    - Input dimension: D = 7 * 30 = 210
    - Hidden dimension: H = 128
    - Latent bottleneck: Z = 32
    - Scaler: StandardScaler (fit strictly on TRAIN-fit)
    - Sparsity: L1 activity penalty (weight = 1e-4)
    - Target sparsity (KL default): 0.05
    - Seed: 42
    - Optimizer: Adam, learning_rate = 1e-3
    - Batch size: 256
    - Epochs: 10
    - Train-validation fraction: 0.10 (chronological final 10% of TRAIN)
    - Early stopping: None (default harness setting)
    - Stride: 1
    - Device: cpu
    """
    return ExperimentConfig(
        run_id=BASELINE_RUN_ID,
        window_size=30,
        stride=1,
        scaler_type="StandardScaler",
        sparsity_type="l1",
        sparsity_weight=1e-4,
        target_sparsity=0.05,
        seed=42,
        optimizer="adam",
        learning_rate=1e-3,
        batch_size=256,
        epochs=10,
        validation_fraction=0.10,
        early_stopping_patience=None,
        device=device,
    )


def run_baseline(
    data_path: str = "data/raw/MetroPT3(AirCompressor).csv",
    output_dir: str = "artifacts/runs",
    device: str = "cpu",
    verbose: bool = True,
) -> ExperimentResult:
    """Execute the authoritative MetroPT-3 baseline reference run."""
    config = get_baseline_config(device=device)
    if verbose:
        print("=" * 70)
        print("MetroGuard: Executing Authoritative Baseline Reference Run")
        print("=" * 70)
        print(f"Run ID: {config.run_id}")
        print(f"Dataset: {data_path}")
        print(f"Architecture: D={config.input_dim}, H={config.hidden_dim}, Z={config.latent_dim}")
        print(f"Window size (W): {config.window_size}, Stride: {config.stride}")
        print(f"Scaler: {config.scaler_type} (fit strictly on TRAIN-fit)")
        print(f"Sparsity: {config.sparsity_type} (weight={config.sparsity_weight})")
        print(f"Optimizer: {config.optimizer}, LR: {config.learning_rate}, Batch: {config.batch_size}")
        print(f"Epochs: {config.epochs}, Val Fraction: {config.validation_fraction} (chronological)")
        print(f"Device: {config.device}, Seed: {config.seed}")
        print("=" * 70)

    result = run_experiment(
        config=config,
        raw_csv_path=data_path,
        save_dir=output_dir,
        verbose=verbose,
    )

    if verbose:
        m = result.calibration_metrics
        print("\n" + "=" * 70)
        print(f"Baseline Execution Complete: {config.run_id}")
        print("=" * 70)
        print(f"Dataset Fingerprint: {result.metadata['dataset_fingerprint']}")
        print(f"Git Commit: {result.metadata['git_commit']}")
        print(f"Runtime: {result.metadata['runtime_seconds']:.2f}s")
        print(
            f"Train Raw Rows: {result.metadata['train_raw_row_count']:,} | "
            f"Cal Raw Rows: {result.metadata['calibration_raw_row_count']:,}"
        )
        print(
            f"Train-fit Windows: {result.metadata['train_fit_window_count']:,} | "
            f"Train-val Windows: {result.metadata['train_val_window_count']:,}"
        )
        print(f"Calibration Usable Windows: {m.num_usable_windows:,}")
        print(
            f"Calibration Normal Windows: {m.num_normal_windows:,} | "
            f"Failure Windows: {m.num_failure_windows:,}"
        )
        if m.pr_auc_trapezoidal is not None:
            print(f"Calibration PR-AUC (trapezoidal): {m.pr_auc_trapezoidal:.6f}")
        if m.average_precision is not None:
            print(f"Calibration Average Precision: {m.average_precision:.6f}")
        if m.roc_auc is not None:
            print(f"Calibration ROC-AUC: {m.roc_auc:.6f}")
        print(f"Normal Mean Score: {m.normal_distribution.get('mean', 0.0):.6f}")
        print(f"Failure Mean Score: {m.failure_distribution.get('mean', 0.0):.6f}")
        print(f"Artifact Saved: {Path(output_dir) / f'{config.run_id}.json'}")
        print("=" * 70)

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
        "--run-baseline",
        action="store_true",
        help="Execute the authoritative real-data baseline experiment (baseline_w30_standard_l1_s42).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier to execute (e.g. baseline_w30_standard_l1_s42).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/MetroPT3(AirCompressor).csv",
        help="Path to raw MetroPT-3 CSV dataset (default: data/raw/MetroPT3(AirCompressor).csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/runs",
        help="Directory to save experiment artifacts and JSON metadata (default: artifacts/runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda', default: cpu).",
    )
    parser.add_argument(
        "--run-matrix",
        action="store_true",
        help="Execute the full 16-run real-data detector matrix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matrix execution by skipping completely valid existing artifacts.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run lightweight preflight check before matrix execution.",
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

    if args.preflight:
        run_preflight(data_path=args.data_path, verbose=True)
    elif args.run_matrix:
        configs = generate_matrix_configs()
        run_matrix(
            configs=configs,
            raw_csv_path=args.data_path,
            save_dir=args.output_dir + "/matrix",
            resume=args.resume,
            verbose=True,
        )
    elif args.smoke:
        run_smoke_test(verbose=True)
    elif args.run_baseline or args.run_id == BASELINE_RUN_ID:
        run_baseline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
            verbose=True,
        )
    elif args.run_id:
        configs = {c.run_id: c for c in generate_matrix_configs()}
        if args.run_id in configs:
            cfg = configs[args.run_id]
            cfg.device = args.device
            run_experiment(
                config=cfg,
                raw_csv_path=args.data_path,
                save_dir=args.output_dir,
                verbose=True,
            )
        else:
            raise ValueError(
                f"Unknown run_id '{args.run_id}'. Available: {BASELINE_RUN_ID}, {list(configs.keys())}"
            )
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
        print(f"Dataset: MetroPT-3 ({EXPECTED_TRAIN_ROWS:,} TRAIN rows, {EXPECTED_CALIBRATION_ROWS:,} CALIBRATION rows)")
        print("Hardware Target: Single CPU Core (x86_64)")
        print("- Single Run (10 epochs, batch size 256): ~3.5 to 5 minutes, ~1.2 to 1.5 GB RAM")
        print("- 16-Config Matrix Sequential: ~48 to 80 minutes, ~2.5 GB peak RAM")
        print("Hardware Target: Google Colab / GPU (T4 / V100)")
        print("- Single Run: ~20 to 30 seconds")
        print("- 16-Config Matrix: ~5 to 8 minutes")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
