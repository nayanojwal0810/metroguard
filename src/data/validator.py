"""Reusable data validation routines for schema, timestamps, missingness, and splits."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from src.data.contract import (
    MAX_GAP_SECONDS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    TIMESTAMP_COL,
    PartitionBoundary,
    SplitPartition,
)


@dataclass
class ValidationReport:
    """Consolidated summary of dataset validation results."""

    is_valid: bool
    row_count: int
    missing_columns: List[str]
    is_timestamp_monotonic: bool
    duplicate_timestamp_count: int
    missing_value_counts: Dict[str, int]
    non_numeric_columns: List[str]
    gap_count: int
    split_counts: Dict[str, int]
    errors: List[str]


def validate_schema(
    df: pd.DataFrame,
    required_features: Sequence[str] = PRIMARY_FEATURES,
    require_timestamp: bool = True,
) -> List[str]:
    """Check that all required feature columns and timestamp exist in DataFrame."""
    missing: List[str] = []
    if require_timestamp and TIMESTAMP_COL not in df.columns:
        missing.append(TIMESTAMP_COL)
    for feat in required_features:
        if feat not in df.columns:
            missing.append(feat)
    return missing


def validate_timestamp_monotonicity(timestamps: pd.Series) -> bool:
    """Verify that timestamps are strictly non-decreasing."""
    ts = pd.to_datetime(timestamps)
    diffs = ts.diff().dropna()
    return bool((diffs >= pd.Timedelta(seconds=0)).all())


def check_duplicate_timestamps(timestamps: pd.Series) -> int:
    """Count duplicate timestamp occurrences."""
    ts = pd.to_datetime(timestamps)
    return int(ts.duplicated().sum())


def validate_numeric_types(
    df: pd.DataFrame,
    features: Sequence[str] = PRIMARY_FEATURES,
) -> List[str]:
    """Ensure all specified feature columns are numeric."""
    non_numeric: List[str] = []
    for feat in features:
        if feat in df.columns and not pd.api.types.is_numeric_dtype(df[feat]):
            non_numeric.append(feat)
    return non_numeric


def validate_missingness(
    df: pd.DataFrame,
    features: Sequence[str] = PRIMARY_FEATURES,
) -> Dict[str, int]:
    """Compute count of NaN / Null values per specified feature."""
    counts: Dict[str, int] = {}
    for feat in features:
        if feat in df.columns:
            counts[feat] = int(df[feat].isna().sum())
    return counts


def detect_gaps(
    timestamps: pd.Series,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> pd.DataFrame:
    """Identify consecutive observations exceeding max_gap_seconds.

    Returns DataFrame with gap_index, prev_timestamp, curr_timestamp, and delta_seconds.
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    deltas = ts.diff().dt.total_seconds()
    gap_mask = deltas > max_gap_seconds

    gap_indices = gap_mask[gap_mask].index
    if len(gap_indices) == 0:
        return pd.DataFrame(
            columns=["gap_index", "prev_timestamp", "curr_timestamp", "delta_seconds"]
        )

    records = []
    for idx in gap_indices:
        records.append(
            {
                "gap_index": idx,
                "prev_timestamp": ts.iloc[idx - 1],
                "curr_timestamp": ts.iloc[idx],
                "delta_seconds": deltas.iloc[idx],
            }
        )
    return pd.DataFrame(records)


def assign_split(timestamps: pd.Series) -> pd.Series:
    """Assign each timestamp to a SplitPartition based on chronological boundaries."""
    ts = pd.to_datetime(timestamps)
    partition_series = pd.Series(
        SplitPartition.OUT_OF_BOUNDS.value,
        index=timestamps.index,
        dtype=object,
    )

    for boundary in SPLIT_BOUNDS:
        mask = (ts >= boundary.start) & (ts <= boundary.end)
        partition_series[mask] = boundary.name.value

    return partition_series


def validate_splits(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL) -> Dict[str, int]:
    """Verify split membership counts and check for unassigned observations."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}'")

    splits = assign_split(df[timestamp_col])
    counts: Dict[str, int] = splits.value_counts().to_dict()
    return counts


def validate_dataset(
    df: pd.DataFrame,
    features: Sequence[str] = PRIMARY_FEATURES,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> ValidationReport:
    """Run full suite of data integrity and schema validation checks."""
    errors: List[str] = []

    # 1. Schema check
    missing_cols = validate_schema(df, required_features=features)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Timestamp checks
    is_monotonic = True
    dup_count = 0
    gap_count = 0
    split_counts: Dict[str, int] = {}

    if TIMESTAMP_COL in df.columns:
        is_monotonic = validate_timestamp_monotonicity(df[TIMESTAMP_COL])
        if not is_monotonic:
            errors.append("Timestamps are not strictly non-decreasing.")

        dup_count = check_duplicate_timestamps(df[TIMESTAMP_COL])
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate timestamps.")

        gaps = detect_gaps(df[TIMESTAMP_COL], max_gap_seconds=max_gap_seconds)
        gap_count = len(gaps)

        split_counts = validate_splits(df, timestamp_col=TIMESTAMP_COL)
        if split_counts.get(SplitPartition.OUT_OF_BOUNDS.value, 0) > 0:
            errors.append(
                f"Found {split_counts[SplitPartition.OUT_OF_BOUNDS.value]} rows outside defined splits."
            )

    # 3. Numeric checks
    non_numeric = validate_numeric_types(df, features=features)
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")

    # 4. Missingness checks
    missing_counts = validate_missingness(df, features=features)
    for feat, count in missing_counts.items():
        if count > 0:
            errors.append(f"Column '{feat}' has {count} missing values.")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        row_count=len(df),
        missing_columns=missing_cols,
        is_timestamp_monotonic=is_monotonic,
        duplicate_timestamp_count=dup_count,
        missing_value_counts=missing_counts,
        non_numeric_columns=non_numeric,
        gap_count=gap_count,
        split_counts=split_counts,
        errors=errors,
    )
