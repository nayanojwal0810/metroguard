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
    event_ids = np.array(["normal"] * len(scores))

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
    event_ids = np.array(["normal"] * len(scores))

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

import json
from pathlib import Path
from unittest.mock import MagicMock
from src.experiments.alert_calibration import verify_replay_consistency
from src.experiments.runner import ExperimentResult
from src.experiments.evaluator import CalibrationMetrics

def setup_mock_gate(monkeypatch, tmp_path):
    # Create mock reference JSON
    ref_data = {
        "calibration_metrics": {
            "average_precision": 0.5,
            "roc_auc": 0.9,
            "event_distributions": {
                "Event_1": {"average_precision": 0.3},
                "Event_2": {"average_precision": 0.4}
            },
            "num_usable_windows": 1000,
            "window_accounting": {"a": 1}
        },
        "metadata": {
            "dataset_fingerprint": "mock_fp",
            "dataset_row_count": 100,
            "train_raw_row_count": 40,
            "calibration_raw_row_count": 60,
            "train_fit_raw_row_count": 35,
            "train_fit_window_count": 30,
            "train_val_raw_row_count": 5,
            "train_val_window_count": 4,
            "split_boundaries": {"train": "mock"},
            "holdout_observations_retained": False,
            "holdout_accessed": False
        },
        "config": {
            "window_size": 180, "stride": 1, "scaler_type": "StandardScaler",
            "sparsity_type": "l1", "sparsity_weight": 0.0001, "hidden_dim": 512,
            "latent_dim": 128, "seed": 42, "optimizer": "adam", "learning_rate": 0.001,
            "batch_size": 256, "epochs": 10, "validation_fraction": 0.1, "features": ["F1"]
        }
    }

    ref_path = tmp_path / "w180_standard_l1_s42.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(ref_data, f)

    def mock_path(path_str):
        if "w180_standard_l1_s42.json" in path_str:
            return ref_path
        return Path(path_str)

    monkeypatch.setattr("src.experiments.alert_calibration.Path", mock_path)

    # Create mock ExperimentResult
    res = MagicMock(spec=ExperimentResult)
    res.metadata = dict(ref_data["metadata"])

    class MockConfig:
        def to_dict(self):
            return dict(ref_data["config"])
    res.config = MockConfig()

    class MockMetrics:
        def to_dict(self):
            return dict(ref_data["calibration_metrics"])
    res.calibration_metrics = MockMetrics()

    return res

def test_gate_exact_match(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    verify_replay_consistency(res, 42)  # Should pass

def test_gate_metrics_mismatch(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    # numerical mismatch beyond tolerance
    res.calibration_metrics.to_dict = lambda: {
        **setup_mock_gate(monkeypatch, tmp_path).calibration_metrics.to_dict(),
        "average_precision": 0.50002
    }
    with pytest.raises(AssertionError, match="Replay Consistency Failed"):
        verify_replay_consistency(res, 42)

def test_gate_config_mismatch(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    class MockConfigBad:
        def to_dict(self):
            cfg = dict(setup_mock_gate(monkeypatch, tmp_path).config.to_dict())
            cfg["learning_rate"] = 0.005
            return cfg
    res.config = MockConfigBad()
    with pytest.raises(AssertionError, match="Config mismatch"):
        verify_replay_consistency(res, 42)

def test_gate_provenance_mismatch(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    res.metadata["train_raw_row_count"] = 999
    with pytest.raises(AssertionError):
        verify_replay_consistency(res, 42)

def test_gate_holdout_access_mismatch(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    res.metadata["holdout_accessed"] = True
    with pytest.raises(AssertionError):
        verify_replay_consistency(res, 42)

from src.experiments.alert_calibration import _normalize_seq

def test_normalize_seq_identical_list_vs_tuple():
    assert _normalize_seq(["a", "b"]) == _normalize_seq(("a", "b"))

def test_normalize_seq_reordered_fails():
    assert _normalize_seq(["a", "b"]) != _normalize_seq(("b", "a"))

def test_normalize_seq_missing_element_fails():
    assert _normalize_seq(["a", "b"]) != _normalize_seq(("a",))

def test_normalize_seq_extra_element_fails():
    assert _normalize_seq(["a", "b"]) != _normalize_seq(("a", "b", "c"))

def test_normalize_seq_changed_scalar_fails():
    assert _normalize_seq(["a", 1]) != _normalize_seq(("a", 2))

def test_normalize_seq_nested_sequence():
    assert _normalize_seq([("a", ["b", "c"]), "d"]) == _normalize_seq([["a", ("b", "c")], "d"])

def test_gate_missing_required_config_field(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    class MockConfigBad:
        def to_dict(self):
            cfg = dict(setup_mock_gate(monkeypatch, tmp_path).config.to_dict())
            del cfg["features"]
            return cfg
    res.config = MockConfigBad()
    with pytest.raises(KeyError):
        verify_replay_consistency(res, 42)

def test_score_trace_non_empty_extraction_and_alignment(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    # Give res some fake scores with 'normal' and 'Event_1'
    res.cal_result = MagicMock()
    res.cal_result.scores = np.array([0.1, 0.2, 5.0, 0.3])
    base_time = pd.Timestamp("2020-04-17 00:00:00")
    res.cal_result.timestamps = pd.DatetimeIndex([base_time + pd.Timedelta(seconds=10*i) for i in range(4)])
    res.cal_result.labels = np.array([0, 0, 1, 0])
    res.cal_result.event_ids = np.array(["normal", "normal", "Event_1", "normal"])

    # Run the sweep just to extract and verify inside the function
    # Wait, run_sweep writes to file.
    # Let's mock calculate_thresholds to capture what it receives
    import src.experiments.alert_calibration
    captured_normal_scores = None
    import src.experiments.alert_calibration
    original_calc = src.experiments.alert_calibration.calculate_thresholds
    def mock_calc(scores):
        nonlocal captured_normal_scores
        captured_normal_scores = scores
        return {95.0: 1.0}

    monkeypatch.setattr("src.experiments.alert_calibration.calculate_thresholds", mock_calc)
    monkeypatch.setattr("src.experiments.alert_calibration.run_experiment", lambda **kw: res)

    from src.experiments.alert_calibration import run_sweep
    run_sweep(base_run_id="w180_standard_l1_s42", seeds=[42], output_dir=str(tmp_path), verbose=False)

    # A. Non-empty extraction
    assert captured_normal_scores is not None
    assert len(captured_normal_scores) == 3

    # B. Alignment (by checking values)
    assert np.allclose(captured_normal_scores, [0.1, 0.2, 0.3])

    # D. Count conservation (already enforced by the assert in the code)

    # E. Chronology (the input was chronological, so the subset should be)
    assert list(captured_normal_scores) == [0.1, 0.2, 0.3]

def test_score_trace_empty_normal_protection(monkeypatch, tmp_path):
    res = setup_mock_gate(monkeypatch, tmp_path)
    res.cal_result = MagicMock()
    # ALL FAILURE!
    res.cal_result.scores = np.array([5.0, 6.0])
    base_time = pd.Timestamp("2020-04-17 00:00:00")
    res.cal_result.timestamps = pd.DatetimeIndex([base_time, base_time + pd.Timedelta(seconds=10)])
    res.cal_result.labels = np.array([1, 1])
    res.cal_result.event_ids = np.array(["Event_1", "Event_1"])

    monkeypatch.setattr("src.experiments.alert_calibration.run_experiment", lambda **kw: res)
    from src.experiments.alert_calibration import run_sweep

    with pytest.raises(ValueError, match="no normal windows"):
        run_sweep(base_run_id="w180_standard_l1_s42", seeds=[42], output_dir=str(tmp_path), verbose=False)

def test_gap_semantics_remain_visible():
    # F. Gap semantics: A >60-second timestamp gap remains visible to the alert episode logic.
    from src.experiments.alert_calibration import compute_alert_episodes
    scores = np.array([10.0, 10.0, 10.0])
    # Gap > 60s between idx 1 and 2
    timestamps = pd.DatetimeIndex([
        pd.Timestamp("2020-04-17 00:00:00"),
        pd.Timestamp("2020-04-17 00:00:10"),
        pd.Timestamp("2020-04-17 00:01:20") # 70 seconds later
    ])
    labels = np.array([0, 0, 0])
    event_ids = np.array(["normal", "normal", "normal"])

    metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, threshold=5.0, persistence=3)
    # Since there's a gap, the streak resets. Thus 0 alerts with persistence=3.
    assert metrics["total_alerts"] == 0

