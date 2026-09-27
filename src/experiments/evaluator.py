"""Calibration evaluation component for Sparse Autoencoder baseline models.

Computes threshold-free calibration metrics:
- Reconstruction error per window
- Descriptive distributions for normal vs documented failure periods
- PR-AUC and ROC-AUC where labels permit
- Usable vs rejected window accounting
- Preserves window timestamps and event associations without updating weights or scalers
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.data.contract import (
    FAILURE_EVENTS,
    FailureEvent,
    SplitPartition,
)
from src.data.windowing import WindowAccounting
from src.models.sae import SparseAutoencoder
from src.preprocessing.scalers import BaseScaler, MinMaxScaler, StandardScaler


@dataclass
class CalibrationMetrics:
    """Threshold-free performance and distribution metrics on CALIBRATION data."""

    num_usable_windows: int
    num_normal_windows: int
    num_failure_windows: int
    failure_ratio: float
    window_accounting: Dict[str, int]
    roc_auc: Optional[float]
    pr_auc_trapezoidal: Optional[float]
    average_precision: Optional[float]
    normal_distribution: Dict[str, float]
    failure_distribution: Dict[str, float]
    event_distributions: Dict[str, Dict[str, float]]
    score_quantiles: Dict[str, float]
    labeling_convention: str = "window_end_timestamp"
    pr_auc: Optional[float] = None  # Backward-compatible alias for pr_auc_trapezoidal

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return asdict(self)


@dataclass
class CalibrationResult:
    """Full calibration evaluation output including arrays and summary metrics."""

    metrics: CalibrationMetrics
    scores: np.ndarray  # Shape: (N,) reconstruction error per window
    labels: np.ndarray  # Shape: (N,) binary ground truth (0: normal, 1: failure)
    event_ids: np.ndarray  # Shape: (N,) event tag or 'normal'
    timestamps: np.ndarray  # Shape: (N,) timestamp for each window end


def compute_distribution_stats(scores: np.ndarray) -> Dict[str, float]:
    """Calculate summary statistics for a 1D array of scores."""
    if len(scores) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=0)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def compute_quantiles(scores: np.ndarray) -> Dict[str, float]:
    """Compute score quantiles across the entire calibration dataset."""
    if len(scores) == 0:
        return {}

    return {
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "p99.9": float(np.percentile(scores, 99.9)),
    }


class CalibrationEvaluator:
    """Evaluates fitted Sparse Autoencoder on CALIBRATION partition without parameter updates."""

    def __init__(
        self,
        model: SparseAutoencoder,
        scaler: BaseScaler,
        failure_events: Optional[Sequence[FailureEvent]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.device = torch.device(device)

        if failure_events is None:
            self.failure_events = tuple(
                e for e in FAILURE_EVENTS if e.partition == SplitPartition.CALIBRATION
            )
        else:
            self.failure_events = tuple(failure_events)

    def assign_event_labels(
        self,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map window evaluation timestamps to binary labels and failure event identifiers.

        Authoritative Labeling Convention:
        A window is labelled strictly according to its evaluation/end timestamp t:
        - If event.start <= t <= event.end: label = 1 (failure), event_id = event.event_id
        - If t < event.start (before event): label = 0 (normal), event_id = 'normal'
        - If t > event.end (after event): label = 0 (normal), event_id = 'normal'
        The window's historical context [t - W + 1, t - 1] does not alter this label.

        Args:
            timestamps: Array of datetime64 or pd.Timestamp values representing
                window end evaluation points.

        Returns:
            Tuple of (labels, event_ids) where labels is 1D int array and event_ids is 1D string array.
        """
        n = len(timestamps)
        labels = np.zeros(n, dtype=np.int32)
        event_ids = np.full(n, "normal", dtype=object)

        if n == 0:
            return labels, event_ids

        ts_series = pd.to_datetime(timestamps)

        for event in self.failure_events:
            in_event = (ts_series >= event.start) & (ts_series <= event.end)
            mask = np.asarray(in_event, dtype=bool)
            labels[mask] = 1
            event_ids[mask] = event.event_id

        return labels, event_ids

    def evaluate(
        self,
        cal_windows: np.ndarray,
        cal_timestamps: np.ndarray,
        window_accounting: Optional[Union[WindowAccounting, Dict[str, int]]] = None,
        num_calibration_rows: Optional[int] = None,
        batch_size: int = 512,
    ) -> CalibrationResult:
        """Run calibration evaluation over windows.

        Ensures:
        - Scaler parameters are not updated.
        - Model weights are not updated.
        - Timestamps and event IDs are preserved.
        - Threshold-free ranking metrics (PR-AUC, ROC-AUC) are calculated.

        Args:
            cal_windows: 3D numpy array of shape (N, W, D_features) or 2D (N, D_flat).
            cal_timestamps: 1D array of window end timestamps, length N.
            window_accounting: WindowAccounting object or dictionary from window builder.
            num_calibration_rows: Total raw observation rows in calibration partition (fallback).
            batch_size: Batch size for model inference.

        Returns:
            CalibrationResult containing scores, ground truth, and CalibrationMetrics.
        """
        if len(cal_windows) != len(cal_timestamps):
            raise ValueError(
                f"cal_windows length ({len(cal_windows)}) must match "
                f"cal_timestamps length ({len(cal_timestamps)})"
            )

        n_usable = len(cal_windows)

        if window_accounting is not None:
            if hasattr(window_accounting, "to_dict"):
                accounting_dict = window_accounting.to_dict()
            else:
                accounting_dict = dict(window_accounting)
        else:
            n_raw = num_calibration_rows if num_calibration_rows is not None else n_usable
            accounting_dict = {
                "num_observations": n_raw,
                "num_candidate_endpoints": n_raw,
                "num_usable_windows": n_usable,
                "num_excluded_insufficient_history": 0,
                "num_excluded_service_gap": 0,
                "num_excluded_split_boundary": 0,
                "num_excluded_out_of_bounds": 0,
            }

        if n_usable == 0:
            empty_metrics = CalibrationMetrics(
                num_usable_windows=0,
                num_normal_windows=0,
                num_failure_windows=0,
                failure_ratio=0.0,
                window_accounting=accounting_dict,
                roc_auc=None,
                pr_auc_trapezoidal=None,
                average_precision=None,
                normal_distribution=compute_distribution_stats(np.array([])),
                failure_distribution=compute_distribution_stats(np.array([])),
                event_distributions={},
                score_quantiles={},
                labeling_convention="window_end_timestamp",
                pr_auc=None,
            )
            return CalibrationResult(
                metrics=empty_metrics,
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int32),
                event_ids=np.empty((0,), dtype=object),
                timestamps=cal_timestamps,
            )

        # 1. Transform windows using fitted TRAIN scaler without updating scaler parameters
        # Capture snapshot of scaler parameters to verify invariance
        scaler_state_before = self._snapshot_scaler()

        if cal_windows.ndim == 3:
            n_w, w_len, n_feat = cal_windows.shape
            reshaped = cal_windows.reshape(-1, n_feat)
            scaled_flat = self.scaler.transform(reshaped)
            scaled_windows = scaled_flat.reshape(n_w, w_len * n_feat)
        elif cal_windows.ndim == 2:
            scaled_windows = self.scaler.transform(cal_windows)
        else:
            raise ValueError(
                f"cal_windows must have 2 or 3 dimensions, got {cal_windows.ndim}"
            )

        # Verify scaler parameters remained identical
        self._verify_scaler_unchanged(scaler_state_before)

        # 2. Compute reconstruction error without gradient tracking
        self.model.eval()
        scores_list: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, n_usable, batch_size):
                batch_arr = scaled_windows[i : i + batch_size]
                batch_tensor = torch.from_numpy(batch_arr).float().to(self.device)
                output = self.model(batch_tensor)
                # sample_losses is 1D tensor of MSE reconstruction error per window
                scores_list.append(output.sample_losses.cpu().numpy())

        scores = np.concatenate(scores_list, axis=0).astype(np.float64)

        # 3. Assign ground truth event labels
        labels, event_ids = self.assign_event_labels(cal_timestamps)

        # 4. Compute metrics
        normal_mask = labels == 0
        failure_mask = labels == 1

        normal_scores = scores[normal_mask]
        failure_scores = scores[failure_mask]

        n_normal = int(np.sum(normal_mask))
        n_failure = int(np.sum(failure_mask))
        failure_ratio = float(n_failure / n_usable) if n_usable > 0 else 0.0

        # PR-AUC (trapezoidal), Average Precision, and ROC-AUC
        pr_auc_trapezoidal: Optional[float] = None
        average_precision: Optional[float] = None
        roc_auc: Optional[float] = None

        if n_failure > 0 and n_normal > 0:
            try:
                from sklearn.metrics import (
                    auc,
                    average_precision_score,
                    precision_recall_curve,
                    roc_auc_score,
                )

                precision, recall, _ = precision_recall_curve(labels, scores)
                pr_auc_trapezoidal = float(auc(recall, precision))
                average_precision = float(average_precision_score(labels, scores))
                roc_auc = float(roc_auc_score(labels, scores))
            except Exception:
                pr_auc_trapezoidal = None
                average_precision = None
                roc_auc = None

        # Event-level breakdown
        event_distributions: Dict[str, Dict[str, float]] = {}
        for event in self.failure_events:
            ev_mask = event_ids == event.event_id
            if np.any(ev_mask):
                event_distributions[event.event_id] = compute_distribution_stats(
                    scores[ev_mask]
                )

        metrics = CalibrationMetrics(
            num_usable_windows=n_usable,
            num_normal_windows=n_normal,
            num_failure_windows=n_failure,
            failure_ratio=failure_ratio,
            window_accounting=accounting_dict,
            roc_auc=roc_auc,
            pr_auc_trapezoidal=pr_auc_trapezoidal,
            average_precision=average_precision,
            normal_distribution=compute_distribution_stats(normal_scores),
            failure_distribution=compute_distribution_stats(failure_scores),
            event_distributions=event_distributions,
            score_quantiles=compute_quantiles(scores),
            labeling_convention="window_end_timestamp",
            pr_auc=pr_auc_trapezoidal,
        )

        return CalibrationResult(
            metrics=metrics,
            scores=scores,
            labels=labels,
            event_ids=event_ids,
            timestamps=cal_timestamps,
        )

    def _snapshot_scaler(self) -> Dict[str, Any]:
        """Capture parameter snapshot of fitted scaler."""
        if isinstance(self.scaler, StandardScaler):
            return {
                "mean_": None if self.scaler.mean_ is None else self.scaler.mean_.copy(),
                "std_": None if self.scaler.std_ is None else self.scaler.std_.copy(),
            }
        elif isinstance(self.scaler, MinMaxScaler):
            return {
                "min_": None if self.scaler.min_ is None else self.scaler.min_.copy(),
                "max_": None if self.scaler.max_ is None else self.scaler.max_.copy(),
                "range_": None if self.scaler.range_ is None else self.scaler.range_.copy(),
            }
        return {}

    def _verify_scaler_unchanged(self, snapshot: Dict[str, Any]) -> None:
        """Assert that scaler parameters have not drifted during transform."""
        if isinstance(self.scaler, StandardScaler):
            if snapshot["mean_"] is not None:
                assert np.array_equal(self.scaler.mean_, snapshot["mean_"]), (
                    "Scaler mean_ changed during calibration evaluation!"
                )
            if snapshot["std_"] is not None:
                assert np.array_equal(self.scaler.std_, snapshot["std_"]), (
                    "Scaler std_ changed during calibration evaluation!"
                )
        elif isinstance(self.scaler, MinMaxScaler):
            if snapshot["min_"] is not None:
                assert np.array_equal(self.scaler.min_, snapshot["min_"]), (
                    "Scaler min_ changed during calibration evaluation!"
                )
            if snapshot["max_"] is not None:
                assert np.array_equal(self.scaler.max_, snapshot["max_"]), (
                    "Scaler max_ changed during calibration evaluation!"
                )
            if snapshot["range_"] is not None:
                assert np.array_equal(self.scaler.range_, snapshot["range_"]), (
                    "Scaler range_ changed during calibration evaluation!"
                )
