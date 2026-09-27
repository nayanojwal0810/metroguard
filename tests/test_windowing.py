"""Unit tests for causal window builder, gap isolation, split isolation, and anti-leakage invariants."""

import numpy as np
import pandas as pd
import pytest

from src.data.contract import PRIMARY_FEATURES, SplitPartition
from src.data.windowing import CausalWindowBuilder


def _create_mock_df(timestamps, values=None):
    """Helper to create test DataFrame with valid primary features."""
    n = len(timestamps)
    data = {"timestamp": timestamps}
    for feat in PRIMARY_FEATURES:
        if values is not None and feat in values:
            data[feat] = values[feat]
        else:
            data[feat] = np.arange(n, dtype=np.float32)
    return pd.DataFrame(data)


def test_causal_window_basic_shape_and_stride():
    """Verify window dimensions, observation count, and stride progression."""
    # 20 consecutive 10s observations in TRAIN
    ts = pd.date_range("2020-02-01 10:00:00", periods=20, freq="10s")
    df = _create_mock_df(ts)

    # Window size 6, stride 1 -> 20 - 6 + 1 = 15 windows
    builder = CausalWindowBuilder(window_size=6, stride=1)
    batch = builder.build_windows(df)

    assert batch.windows.shape == (15, 6, len(PRIMARY_FEATURES))
    assert len(batch.timestamps) == 15
    assert len(batch.partitions) == 15
    assert (batch.partitions == SplitPartition.TRAIN.value).all()
    # Check that the first window ends at timestamp of index 5 (6th observation)
    assert batch.timestamps[0] == pd.Timestamp("2020-02-01 10:00:50")

    # Stride 2 -> 8 windows (indices 5, 7, 9, 11, 13, 15, 17, 19)
    builder_s2 = CausalWindowBuilder(window_size=6, stride=2)
    batch_s2 = builder_s2.build_windows(df)
    assert batch_s2.windows.shape == (8, 6, len(PRIMARY_FEATURES))


def test_windows_do_not_cross_service_gaps():
    """Verify that windows never span across service breaks (delta_t > 60s)."""
    # 5 observations, then a 120-second gap, then 8 observations
    part1 = pd.date_range("2020-02-01 10:00:00", periods=5, freq="10s")
    gap_time = part1[-1] + pd.Timedelta(seconds=120)
    part2 = pd.date_range(gap_time, periods=8, freq="10s")
    ts = pd.concat([pd.Series(part1), pd.Series(part2)], ignore_index=True)
    df = _create_mock_df(ts)

    # With window_size=6:
    # part1 has 5 rows -> 0 windows
    # part2 has 8 rows -> 8 - 6 + 1 = 3 windows
    # Total windows must be exactly 3, and NONE can contain part1 rows
    builder = CausalWindowBuilder(window_size=6, stride=1, max_gap_seconds=60.0)
    batch = builder.build_windows(df)

    assert len(batch.windows) == 3
    # First window in part2 must start at part2[0] and end at part2[5]
    assert batch.timestamps[0] == part2[5]
    # Check that values in windows correspond strictly to part2 rows (indices 5 to 12)
    expected_first_window_tp2 = df["TP2"].iloc[5:11].to_numpy()
    np.testing.assert_array_equal(batch.windows[0, :, 0], expected_first_window_tp2)


def test_windows_do_not_cross_split_boundaries():
    """Verify that windows never cross from TRAIN into CALIBRATION."""
    # 5 observations right before Train end, 5 observations right after Calibration start
    # No gap in time (all 10s apart), but crossing partition boundary:
    # Train ends at 2020-03-31 23:59:59; Calibration starts at 2020-04-01 00:00:00
    ts_train = pd.Series(
        [
            pd.Timestamp("2020-03-31 23:59:10"),
            pd.Timestamp("2020-03-31 23:59:20"),
            pd.Timestamp("2020-03-31 23:59:30"),
            pd.Timestamp("2020-03-31 23:59:40"),
            pd.Timestamp("2020-03-31 23:59:50"),
        ]
    )
    ts_cal = pd.Series(
        [
            pd.Timestamp("2020-04-01 00:00:00"),
            pd.Timestamp("2020-04-01 00:00:10"),
            pd.Timestamp("2020-04-01 00:00:20"),
            pd.Timestamp("2020-04-01 00:00:30"),
            pd.Timestamp("2020-04-01 00:00:40"),
            pd.Timestamp("2020-04-01 00:00:50"),
        ]
    )
    ts = pd.concat([ts_train, ts_cal], ignore_index=True)
    df = _create_mock_df(ts)

    builder = CausalWindowBuilder(window_size=6, stride=1)
    batch = builder.build_windows(df)

    # Train only had 5 observations -> 0 windows
    # Calibration has 6 observations -> exactly 1 window
    # No window spans across the boundary
    assert len(batch.windows) == 1
    assert batch.partitions[0] == SplitPartition.CALIBRATION.value
    assert batch.timestamps[0] == pd.Timestamp("2020-04-01 00:00:50")
    # Features must match only calibration rows (indices 5 to 10)
    expected_tp2 = df["TP2"].iloc[5:11].to_numpy()
    np.testing.assert_array_equal(batch.windows[0, :, 0], expected_tp2)


def test_insufficient_history_produces_no_fabricated_padding():
    """Verify that rows with fewer than W preceding observations produce no window."""
    # Only 4 observations when W=6
    ts = pd.date_range("2020-02-01 10:00:00", periods=4, freq="10s")
    df = _create_mock_df(ts)

    builder = CausalWindowBuilder(window_size=6, stride=1)
    batch = builder.build_windows(df)

    assert len(batch.windows) == 0
    assert len(batch.timestamps) == 0


def test_future_row_cannot_enter_past_window():
    """LEAKAGE INVARIANT TEST: Future observations must not alter past window content.

    Demonstrates that modifying, appending, or replacing a future observation at t > t_eval
    has ZERO effect on the window constructed at time t_eval.
    """
    ts = pd.date_range("2020-02-01 10:00:00", periods=10, freq="10s")
    df_original = _create_mock_df(ts)

    builder = CausalWindowBuilder(window_size=6, stride=1)
    batch_orig = builder.build_windows(df_original)

    # Let t_eval be the end of the first window (index 5)
    window_0_orig = batch_orig.windows[0].copy()
    timestamp_0_orig = batch_orig.timestamps[0]
    assert timestamp_0_orig == ts[5]

    # Create modified scenario: alter future rows (indices 6, 7, 8, 9) with extreme values
    df_future_modified = df_original.copy()
    for feat in PRIMARY_FEATURES:
        df_future_modified.loc[6:, feat] = 99999.0

    batch_future_mod = builder.build_windows(df_future_modified)
    window_0_future_mod = batch_future_mod.windows[0]

    # Past window at index 5 must remain 100% IDENTICAL
    np.testing.assert_array_equal(
        window_0_orig,
        window_0_future_mod,
        err_msg="Future data leaked into past window!",
    )

    # Contrast with modifying an IN-WINDOW row (index 4)
    df_past_modified = df_original.copy()
    df_past_modified.loc[4, "TP2"] = 12345.0
    batch_past_mod = builder.build_windows(df_past_modified)
    window_0_past_mod = batch_past_mod.windows[0]

    assert not np.array_equal(window_0_orig, window_0_past_mod), (
        "Window failed to reflect in-window historical change"
    )
