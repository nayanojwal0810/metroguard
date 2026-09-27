"""Causal window construction enforcing gap rules, split boundaries, and zero temporal leakage."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.data.contract import (
    MAX_GAP_SECONDS,
    PRIMARY_FEATURES,
    TIMESTAMP_COL,
    WINDOW_CANDIDATES,
    SplitPartition,
)
from src.data.validator import assign_split


@dataclass(frozen=True)
class WindowAccounting:
    """Explicit deterministic accounting of observations and window endpoints."""

    num_observations: int
    num_candidate_endpoints: int
    num_usable_windows: int
    num_excluded_insufficient_history: int  # Initial sequence or sub-segment before W observations
    num_excluded_service_gap: int          # Service gap (delta_t > 60s) within window
    num_excluded_split_boundary: int       # Window would cross split partition boundary
    num_excluded_out_of_bounds: int        # Observation itself is OUT_OF_BOUNDS

    def to_dict(self) -> Dict[str, int]:
        """Convert accounting to dictionary."""
        return {
            "num_observations": self.num_observations,
            "num_candidate_endpoints": self.num_candidate_endpoints,
            "num_usable_windows": self.num_usable_windows,
            "num_excluded_insufficient_history": self.num_excluded_insufficient_history,
            "num_excluded_service_gap": self.num_excluded_service_gap,
            "num_excluded_split_boundary": self.num_excluded_split_boundary,
            "num_excluded_out_of_bounds": self.num_excluded_out_of_bounds,
        }


@dataclass(frozen=True)
class WindowBatch:
    """Container for constructed causal windows and associated metadata."""

    windows: np.ndarray  # Shape: (N, W, D)
    timestamps: np.ndarray  # Shape: (N,) — timestamp of the latest observation in each window
    partitions: np.ndarray  # Shape: (N,) — split partition for each window
    window_size: int
    stride: int
    feature_names: Tuple[str, ...]
    accounting: Optional[WindowAccounting] = None


class CausalWindowBuilder:
    """Builds causal rolling windows of continuous sensor observations.

    Authoritative Window Contract:
    - A window contains exactly W consecutive observations.
    - The window ends at the evaluation observation t.
    - All observations are at or before t.
    - Every consecutive timestamp gap within the window must be <= max_gap_seconds (default 60s).
    - A window may not cross a train/calibration/holdout boundary.
    - A window may not cross a service gap.
    - No padding, interpolation, or synthetic observations are introduced.
    - Therefore W represents observation count, not an exact elapsed-time duration.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 1,
        max_gap_seconds: float = MAX_GAP_SECONDS,
        features: Sequence[str] = PRIMARY_FEATURES,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if max_gap_seconds <= 0:
            raise ValueError(f"max_gap_seconds must be > 0, got {max_gap_seconds}")

        self.window_size = window_size
        self.stride = stride
        self.max_gap_seconds = max_gap_seconds
        self.features = tuple(features)

    def build_windows(
        self,
        df: pd.DataFrame,
        timestamp_col: str = TIMESTAMP_COL,
    ) -> WindowBatch:
        """Construct causal windows from a sorted DataFrame.

        Args:
            df: DataFrame containing timestamps and feature columns. Must be
                chronologically sorted.
            timestamp_col: Name of the timestamp column.

        Returns:
            WindowBatch containing 3D window array, end timestamps, and split labels.
        """
        if len(df) == 0:
            return WindowBatch(
                windows=np.empty((0, self.window_size, len(self.features)), dtype=np.float32),
                timestamps=np.empty((0,), dtype="datetime64[ns]"),
                partitions=np.empty((0,), dtype=object),
                window_size=self.window_size,
                stride=self.stride,
                feature_names=self.features,
                accounting=WindowAccounting(
                    num_observations=0,
                    num_candidate_endpoints=0,
                    num_usable_windows=0,
                    num_excluded_insufficient_history=0,
                    num_excluded_service_gap=0,
                    num_excluded_split_boundary=0,
                    num_excluded_out_of_bounds=0,
                ),
            )

        # Ensure timestamp is datetime and features are extracted
        ts = pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
        feature_data = df[list(self.features)].to_numpy(dtype=np.float32)

        # Assign split partition labels
        partitions = assign_split(ts).to_numpy()

        # Compute consecutive timestamp deltas in seconds
        deltas = ts.diff().dt.total_seconds().fillna(0.0).to_numpy()

        # Identify boundary resets and classify causes:
        # A reset occurs at index j if:
        # 1. j == 0 (start of series)
        # 2. deltas[j] > max_gap_seconds (service gap)
        # 3. partitions[j] != partitions[j - 1] (split boundary transition)
        # 4. partitions[j] == OUT_OF_BOUNDS (unassigned data)
        n_rows = len(df)
        reset_cause = np.zeros(n_rows, dtype=object)
        reset_cause[0] = "start_of_series"

        is_gap = deltas > self.max_gap_seconds
        reset_cause[is_gap] = "service_gap"

        split_changed = np.zeros(n_rows, dtype=bool)
        split_changed[1:] = partitions[1:] != partitions[:-1]
        reset_cause[split_changed] = "split_boundary"

        is_oob = partitions == SplitPartition.OUT_OF_BOUNDS.value
        reset_cause[is_oob] = "out_of_bounds"

        reset_mask = reset_cause != 0

        # Calculate segment start index for each row
        # Segment start is the most recent reset index at or before i
        segment_starts = np.zeros(n_rows, dtype=np.int64)
        current_start = 0
        for i in range(n_rows):
            if reset_mask[i]:
                current_start = i
            segment_starts[i] = current_start

        # Explicit candidate endpoint evaluation and exclusion categorization
        window_endpoints = list(range(self.window_size - 1, n_rows, self.stride))
        initial_insufficient = min(n_rows, self.window_size - 1)

        n_insufficient_history = initial_insufficient
        n_service_gap = 0
        n_split_boundary = 0
        n_out_of_bounds = 0
        valid_end_indices: List[int] = []

        for i in window_endpoints:
            if is_oob[i]:
                n_out_of_bounds += 1
                continue

            start_idx = i - self.window_size + 1
            if start_idx < segment_starts[i]:
                cause = reset_cause[segment_starts[i]]
                if cause == "service_gap":
                    n_service_gap += 1
                elif cause == "split_boundary":
                    n_split_boundary += 1
                elif cause == "out_of_bounds":
                    n_out_of_bounds += 1
                else:
                    n_insufficient_history += 1
            else:
                valid_end_indices.append(i)

        n_candidate = len(window_endpoints) + initial_insufficient
        accounting = WindowAccounting(
            num_observations=n_rows,
            num_candidate_endpoints=n_candidate,
            num_usable_windows=len(valid_end_indices),
            num_excluded_insufficient_history=n_insufficient_history,
            num_excluded_service_gap=n_service_gap,
            num_excluded_split_boundary=n_split_boundary,
            num_excluded_out_of_bounds=n_out_of_bounds,
        )

        if not valid_end_indices:
            return WindowBatch(
                windows=np.empty((0, self.window_size, len(self.features)), dtype=np.float32),
                timestamps=np.empty((0,), dtype="datetime64[ns]"),
                partitions=np.empty((0,), dtype=object),
                window_size=self.window_size,
                stride=self.stride,
                feature_names=self.features,
                accounting=accounting,
            )

        # Slice 3D array of windows: shape (num_windows, window_size, num_features)
        n_windows = len(valid_end_indices)
        windows = np.empty(
            (n_windows, self.window_size, len(self.features)),
            dtype=np.float32,
        )
        for out_idx, end_idx in enumerate(valid_end_indices):
            start_idx = end_idx - self.window_size + 1
            windows[out_idx] = feature_data[start_idx : end_idx + 1]

        end_indices_arr = np.array(valid_end_indices, dtype=np.int64)
        window_timestamps = ts.iloc[end_indices_arr].to_numpy()
        window_partitions = partitions[end_indices_arr]

        return WindowBatch(
            windows=windows,
            timestamps=window_timestamps,
            partitions=window_partitions,
            window_size=self.window_size,
            stride=self.stride,
            feature_names=self.features,
            accounting=accounting,
        )
