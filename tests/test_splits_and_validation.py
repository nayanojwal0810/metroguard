"""Unit tests for split boundary definitions, schema validation, and gap detection."""

import numpy as np
import pandas as pd
import pytest

from src.data.contract import (
    MAX_GAP_SECONDS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    SplitPartition,
)
from src.data.validator import (
    assign_split,
    check_duplicate_timestamps,
    detect_gaps,
    validate_missingness,
    validate_numeric_types,
    validate_schema,
    validate_splits,
    validate_timestamp_monotonicity,
)


def test_split_bounds_no_overlap():
    """Verify that split boundaries are strictly chronological and non-overlapping."""
    for i in range(len(SPLIT_BOUNDS) - 1):
        curr_b = SPLIT_BOUNDS[i]
        next_b = SPLIT_BOUNDS[i + 1]
        assert curr_b.start <= curr_b.end, f"{curr_b.name} start after end"
        assert curr_b.end < next_b.start, (
            f"Overlap or violation between {curr_b.name} ({curr_b.end}) "
            f"and {next_b.name} ({next_b.start})"
        )


def test_assign_split_membership():
    """Verify exact membership assignment across defined split dates."""
    test_dates = pd.Series(
        [
            pd.Timestamp("2020-02-01 00:00:00"),  # Train start
            pd.Timestamp("2020-03-31 23:59:59"),  # Train end
            pd.Timestamp("2020-04-01 00:00:00"),  # Calibration start
            pd.Timestamp("2020-05-31 23:59:59"),  # Calibration end
            pd.Timestamp("2020-06-01 00:00:00"),  # Holdout start
            pd.Timestamp("2020-08-31 23:59:59"),  # Holdout end
            pd.Timestamp("2020-09-01 00:00:00"),  # Tail start
            pd.Timestamp("2020-09-01 03:59:50"),  # Tail end
            pd.Timestamp("2020-09-01 04:00:00"),  # Beyond tail
            pd.Timestamp("2020-01-31 23:59:59"),  # Before train
        ]
    )
    assigned = assign_split(test_dates)
    expected = [
        SplitPartition.TRAIN.value,
        SplitPartition.TRAIN.value,
        SplitPartition.CALIBRATION.value,
        SplitPartition.CALIBRATION.value,
        SplitPartition.HOLDOUT.value,
        SplitPartition.HOLDOUT.value,
        SplitPartition.UNUSED_TAIL.value,
        SplitPartition.UNUSED_TAIL.value,
        SplitPartition.OUT_OF_BOUNDS.value,
        SplitPartition.OUT_OF_BOUNDS.value,
    ]
    assert list(assigned) == expected


def test_validate_schema():
    """Test schema validation identifies present and missing columns."""
    df_valid = pd.DataFrame(columns=["timestamp"] + list(PRIMARY_FEATURES))
    assert validate_schema(df_valid) == []

    df_invalid = pd.DataFrame(columns=["timestamp", "TP2", "TP3"])
    missing = validate_schema(df_invalid)
    assert "H1" in missing
    assert "Motor_current" in missing


def test_validate_timestamp_monotonicity():
    """Test verification of strictly non-decreasing timestamps."""
    monotonic = pd.Series(pd.date_range("2020-02-01", periods=10, freq="10s"))
    assert validate_timestamp_monotonicity(monotonic) is True

    non_monotonic = monotonic.copy()
    non_monotonic.iloc[5] = pd.Timestamp("2020-01-01")
    assert validate_timestamp_monotonicity(non_monotonic) is False


def test_check_duplicate_timestamps():
    """Test identification of duplicated timestamps."""
    unique = pd.Series(pd.date_range("2020-02-01", periods=10, freq="10s"))
    assert check_duplicate_timestamps(unique) == 0

    with_dups = unique.copy()
    with_dups.iloc[3] = with_dups.iloc[2]
    assert check_duplicate_timestamps(with_dups) == 1


def test_validate_numeric_types():
    """Test numeric column type checking."""
    df = pd.DataFrame(
        {
            "TP2": [1.0, 2.0],
            "TP3": [1.0, "invalid_str"],
        }
    )
    assert validate_numeric_types(df, features=["TP2"]) == []
    assert validate_numeric_types(df, features=["TP2", "TP3"]) == ["TP3"]


def test_validate_missingness():
    """Test missing/null value detection across features."""
    df = pd.DataFrame(
        {
            "TP2": [1.0, np.nan, 3.0],
            "TP3": [1.0, 2.0, 3.0],
        }
    )
    counts = validate_missingness(df, features=["TP2", "TP3"])
    assert counts["TP2"] == 1
    assert counts["TP3"] == 0


def test_detect_gaps():
    """Test detection of service breaks where delta_t > max_gap_seconds."""
    ts = pd.Series(
        [
            pd.Timestamp("2020-02-01 00:00:00"),
            pd.Timestamp("2020-02-01 00:00:10"),
            pd.Timestamp("2020-02-01 00:00:20"),
            pd.Timestamp("2020-02-01 00:02:00"),  # 100s gap
            pd.Timestamp("2020-02-01 00:02:10"),
        ]
    )
    gaps = detect_gaps(ts, max_gap_seconds=MAX_GAP_SECONDS)
    assert len(gaps) == 1
    assert gaps.iloc[0]["delta_seconds"] == 100.0
    assert gaps.iloc[0]["gap_index"] == 3
