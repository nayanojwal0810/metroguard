"""Causal window construction enforcing gap rules, split boundaries, and zero temporal leakage."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
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
class WindowBatch:
    """Container for constructed causal windows and associated metadata."""

    windows: np.ndarray  # Shape: (N, W, D)
    timestamps: np.ndarray  # Shape: (N,) — timestamp of the latest observation in each window
    partitions: np.ndarray  # Shape: (N,) — split partition for each window
    window_size: int
    stride: int
    feature_names: Tuple[str, ...]


class CausalWindowBuilder:
    """Builds causal rolling windows of continuous sensor observations.

    Guarantees:
    1. Strictly causal: Window at index i uses only observations from [i - W + 1, i].
    2. Fixed observation count: Every window contains exactly W observations.
    3. Gap isolation: Windows never cross telemetry breaks where Δt > max_gap_seconds.
    4. Split isolation: Windows never cross chronological split boundaries.
    5. Zero fabrication: Insufficient history produces no window (no padding/backfilling).
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
            )

        # Ensure timestamp is datetime and features are extracted
        ts = pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
        feature_data = df[list(self.features)].to_numpy(dtype=np.float32)

        # Assign split partition labels
        partitions = assign_split(ts).to_numpy()

        # Compute consecutive timestamp deltas in seconds
        deltas = ts.diff().dt.total_seconds().fillna(0.0).to_numpy()

        # Identify boundary resets:
        # A reset occurs at index j if:
        # 1. j == 0 (start of series)
        # 2. deltas[j] > max_gap_seconds (service gap)
        # 3. partitions[j] != partitions[j - 1] (split boundary transition)
        # 4. partitions[j] == OUT_OF_BOUNDS (unassigned data)
        n_rows = len(df)
        reset_mask = np.zeros(n_rows, dtype=bool)
        reset_mask[0] = True

        # Telemetry gaps
        reset_mask[deltas > self.max_gap_seconds] = True

        # Split boundary transitions
        split_changed = partitions[1:] != partitions[:-1]
        reset_mask[1:][split_changed] = True

        # Exclude out-of-bounds rows from forming valid segments
        is_oob = partitions == SplitPartition.OUT_OF_BOUNDS.value
        reset_mask[is_oob] = True

        # Calculate segment start index for each row
        # Segment start is the most recent reset index at or before i
        segment_starts = np.zeros(n_rows, dtype=np.int64)
        current_start = 0
        for i in range(n_rows):
            if reset_mask[i]:
                current_start = i
            segment_starts[i] = current_start

        # A window ending at index i covering [i - W + 1, i] is valid if:
        # 1. i - W + 1 >= segment_starts[i] (no reset within window)
        # 2. partitions[i] != OUT_OF_BOUNDS
        valid_end_indices: List[int] = []
        for i in range(self.window_size - 1, n_rows, self.stride):
            start_idx = i - self.window_size + 1
            if start_idx >= segment_starts[i] and not is_oob[i]:
                valid_end_indices.append(i)

        if not valid_end_indices:
            return WindowBatch(
                windows=np.empty((0, self.window_size, len(self.features)), dtype=np.float32),
                timestamps=np.empty((0,), dtype="datetime64[ns]"),
                partitions=np.empty((0,), dtype=object),
                window_size=self.window_size,
                stride=self.stride,
                feature_names=self.features,
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
        )
