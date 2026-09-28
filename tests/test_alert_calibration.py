import pytest
import numpy as np
import pandas as pd
from datetime import timedelta
from src.experiments.alert_calibration import compute_alert_episodes, calculate_thresholds

def test_threshold_correctness():
    normal_scores = np.arange(101).astype(float)  # 0 to 100
    thresholds = calculate_thresholds(normal_scores)
    assert thresholds[95.0] == 95.0
    assert thresholds[99.0] == 99.0

def test_persistence_correctness_and_deduplication():
    # Scores: 0 0 10 10 10 10 0 10 10 0 0
    # Thresh = 5, Persistence = 3
    scores = np.array([0, 0, 10, 10, 10, 10, 0, 10, 10, 0, 0], dtype=float)
    base_time = pd.Timestamp("2020-04-10 00:00:00")
    timestamps = pd.DatetimeIndex([base_time + timedelta(seconds=10*i) for i in range(len(scores))])
    labels = np.zeros(len(scores))
    event_ids = np.array(["None"] * len(scores))
    
    metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, threshold=5.0, persistence=3)
    # The first 3-streak is at indices 2, 3, 4. The 4th high score (index 5) is part of the same contiguous streak, so it's deduped.
    # The second high score group is at indices 7, 8, which is only length 2 (less than 3). So no alert.
    # Total alerts should be exactly 1.
    assert metrics["total_alerts"] == 1
    assert metrics["normal_alerts"] == 1

def test_gap_reset():
    # Gap > 60s
    scores = np.array([10, 10, 10], dtype=float)
    base_time = pd.Timestamp("2020-04-10 00:00:00")
    # T0, T0+10, T0+80 (gap is 70s)
    timestamps = pd.DatetimeIndex([base_time, base_time + timedelta(seconds=10), base_time + timedelta(seconds=80)])
    labels = np.zeros(len(scores))
    event_ids = np.array(["None"] * len(scores))
    
    metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, threshold=5.0, persistence=3)
    # Gap > 60s breaks the streak
    assert metrics["total_alerts"] == 0

def test_event_timing_and_lead_time():
    # Event 1 starts at 2020-04-18 00:00:00
    scores = np.array([10, 10, 10, 10], dtype=float)
    base_time = pd.Timestamp("2020-04-17 23:59:40")
    # T0, T0+10, T0+20, T0+30
    timestamps = pd.DatetimeIndex([base_time + timedelta(seconds=10*i) for i in range(len(scores))])
    labels = np.zeros(len(scores))
    event_ids = np.array(["Event_1"] * len(scores))
    
    # Alert will happen on the 3rd window: T0 + 20s = 2020-04-18 00:00:00
    metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, threshold=5.0, persistence=3)
    assert metrics["total_alerts"] == 1
    assert metrics["event_1_detected"] == True
    # The first qualifying window was T0 = 23:59:40. Lead time is 20s.
    assert metrics["event_1_lead_time_sec"] == 20.0

    # Let's shift it earlier
    base_time = pd.Timestamp("2020-04-17 23:58:00")
    timestamps = pd.DatetimeIndex([base_time + timedelta(seconds=10*i) for i in range(len(scores))])
    metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, threshold=5.0, persistence=3)
    # Alert at T0 = 23:58:00
    # Event starts at 00:00:00
    # Lead time = 120 seconds
    assert metrics["event_1_lead_time_sec"] == 120.0
