import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time

from src.experiments.runner import run_experiment, get_git_commit
from src.experiments.config import generate_matrix_configs, ExperimentConfig
from src.data.contract import FAILURE_EVENTS

@dataclass
class AlertPolicyRow:
    seed: int
    threshold_percentile: float
    threshold_value: float
    persistence: int

    window_average_precision: float
    window_roc_auc: float
    window_precision: float
    window_recall: float
    window_f1: float

    total_alerts: int
    false_alert_episodes: int
    false_alerts_per_30_days: float
    alerted_duration_sec: float

    event_1_detected: bool
    event_1_first_alert_time: str
    event_1_detection_class: str
    event_1_lead_time_sec: float
    event_1_detection_delay_sec: float

    event_2_detected: bool
    event_2_first_alert_time: str
    event_2_detection_class: str
    event_2_lead_time_sec: float
    event_2_detection_delay_sec: float

def calculate_thresholds(normal_scores: np.ndarray) -> Dict[float, float]:
    percentiles = [95.0, 97.0, 98.0, 99.0, 99.5, 99.7, 99.9, 99.95, 99.99]
    return {p: float(np.percentile(normal_scores, p)) for p in percentiles}

def validate_and_normalize_trace(scores: Any, timestamps: Any, labels: Any, event_ids: Any) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    timestamps = pd.DatetimeIndex(timestamps)
    labels = np.asarray(labels, dtype=int)
    event_ids = np.asarray(event_ids, dtype=object)

    if scores.ndim != 1 or timestamps.ndim != 1 or labels.ndim != 1 or event_ids.ndim != 1:
        raise ValueError("Trace arrays must be 1-dimensional.")

    n = len(scores)
    if not (n == len(timestamps) == len(labels) == len(event_ids)):
        raise ValueError("Score trace arrays have mismatched lengths.")

    if n == 0:
        raise ValueError("Trace arrays cannot be empty.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps are not chronologically ordered.")

    if np.isnan(scores).any() or np.isinf(scores).any():
        raise ValueError("Calibration score trace contains NaN or Inf values.")

    return scores, timestamps, labels, event_ids

def compute_alert_episodes(
    scores: Any,
    timestamps: Any,
    labels: Any,
    event_ids: Any,
    threshold: float,
    persistence: int
) -> Dict[str, Any]:
    from src.experiments.alert_calibration import validate_and_normalize_trace
    scores, timestamps, labels, event_ids = validate_and_normalize_trace(scores, timestamps, labels, event_ids)

    n = len(scores)
    above_threshold = (scores >= threshold)

    # Window metrics
    # Normal windows = 0, Failure = 1
    # We treat >= threshold as predicted 1
    tp = np.sum((above_threshold) & (labels == 1))
    fp = np.sum((above_threshold) & (labels == 0))
    fn = np.sum((~above_threshold) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # We will compute window AP and ROC AUC outside this function per threshold,
    # but wait, AP and ROC AUC are threshold-independent!
    # The requirement says: "For every policy row, expose: window_average_precision, window_roc_auc..."
    # We can just compute them once or pass them in.
    # But wait, sklearn's average_precision_score is for the full ranking.
    # I can compute it or leave it as 0.0 if not needed per threshold?
    # No, I should pass the overall cal metrics into this function.

    # Let's continue with episodes
    time_diffs = np.diff(timestamps.values) / np.timedelta64(1, 's')
    is_gap = np.insert(time_diffs > 60.0, 0, False)

    alerts = []
    current_streak = 0
    in_alert_episode = False

    alerted_windows = 0

    for i in range(n):
        if is_gap[i]:
            current_streak = 0
            in_alert_episode = False

        if above_threshold[i]:
            current_streak += 1
            if current_streak >= persistence and not in_alert_episode:
                alerts.append(i - persistence + 1)
                in_alert_episode = True
            if in_alert_episode:
                alerted_windows += 1
        else:
            current_streak = 0
            in_alert_episode = False

    alerted_duration_sec = alerted_windows * 10.0  # approximate 10s per window

    valid_diffs = time_diffs[time_diffs <= 60.0]
    eligible_duration_sec = valid_diffs.sum() + 10.0 if len(valid_diffs) > 0 else 10.0
    days = eligible_duration_sec / 86400.0

    # Event definitions
    # 12-hour attribution window
    ATTRIBUTION_WINDOW = pd.Timedelta(hours=12)

    # Find events
    events_meta = {}
    for ev in FAILURE_EVENTS:
        events_meta[ev.event_id] = {
            "start": ev.start,
            "end": ev.end,
            "detected": False,
            "first_alert_time": "None",
            "detection_class": "not_detected",
            "lead_time_sec": 0.0,
            "detection_delay_sec": 0.0
        }

    false_alert_episodes = 0

    for idx in alerts:
        t = timestamps[idx]
        attributed = False

        for eid, meta in events_meta.items():
            if meta["detected"]:
                # Already detected, but we still attribute alerts within the event window to it
                # so they don't count as false alerts
                if (t >= meta["start"] - ATTRIBUTION_WINDOW) and (t <= meta["end"] + ATTRIBUTION_WINDOW):
                    attributed = True
                continue

            # Check if alert falls in this event's windows
            pre_event = (t >= meta["start"] - ATTRIBUTION_WINDOW) and (t < meta["start"])
            in_event = (t >= meta["start"]) and (t <= meta["end"])
            post_event = (t > meta["end"]) and (t <= meta["end"] + ATTRIBUTION_WINDOW)

            if pre_event or in_event or post_event:
                meta["detected"] = True
                meta["first_alert_time"] = t.isoformat()
                attributed = True
                if pre_event:
                    meta["detection_class"] = "pre_event"
                    meta["lead_time_sec"] = (meta["start"] - t).total_seconds()
                    meta["detection_delay_sec"] = 0.0
                elif in_event:
                    meta["detection_class"] = "in_event"
                    meta["lead_time_sec"] = 0.0
                    meta["detection_delay_sec"] = (t - meta["start"]).total_seconds()
                else:
                    meta["detection_class"] = "post_event"
                    meta["lead_time_sec"] = 0.0
                    meta["detection_delay_sec"] = (t - meta["start"]).total_seconds()

        if not attributed:
            false_alert_episodes += 1

    return {
        "window_precision": precision,
        "window_recall": recall,
        "window_f1": f1,

        "total_alerts": len(alerts),
        "false_alert_episodes": false_alert_episodes,
        "false_alerts_per_30_days": (false_alert_episodes / days) * 30 if days > 0 else 0.0,
        "alerted_duration_sec": alerted_duration_sec,

        "event_1_detected": events_meta["Event_1"]["detected"] if "Event_1" in events_meta else False,
        "event_1_first_alert_time": events_meta["Event_1"]["first_alert_time"] if "Event_1" in events_meta else "None",
        "event_1_detection_class": events_meta["Event_1"]["detection_class"] if "Event_1" in events_meta else "not_detected",
        "event_1_lead_time_sec": events_meta["Event_1"]["lead_time_sec"] if "Event_1" in events_meta else 0.0,
        "event_1_detection_delay_sec": events_meta["Event_1"]["detection_delay_sec"] if "Event_1" in events_meta else 0.0,

        "event_2_detected": events_meta["Event_2"]["detected"] if "Event_2" in events_meta else False,
        "event_2_first_alert_time": events_meta["Event_2"]["first_alert_time"] if "Event_2" in events_meta else "None",
        "event_2_detection_class": events_meta["Event_2"]["detection_class"] if "Event_2" in events_meta else "not_detected",
        "event_2_lead_time_sec": events_meta["Event_2"]["lead_time_sec"] if "Event_2" in events_meta else 0.0,
        "event_2_detection_delay_sec": events_meta["Event_2"]["detection_delay_sec"] if "Event_2" in events_meta else 0.0,
    }

def _normalize_seq(val: Any) -> Any:
    """
    Recursively normalize tuples to lists to handle JSON serialization representation differences.
    Preserves exact ordering and all other scalar types.
    """
    if isinstance(val, (list, tuple)):
        return [_normalize_seq(x) for x in val]
    if isinstance(val, dict):
        return {k: _normalize_seq(v) for k, v in val.items()}
    return val

def verify_replay_consistency(res: Any, seed: int) -> None:
    # res is an ExperimentResult
    run_id = f"w180_standard_l1_s{seed}"
    if seed == 42:
        ref_path = Path("artifacts/runs/matrix_final_s42/matrix/w180_standard_l1_s42.json")
    elif seed in [7, 21]:
        ref_path = Path(f"artifacts/runs/repeatability_w180_standard_l1/w180_standard_l1_s{seed}.json")
    else:
        raise ValueError(f"No reference artifact defined for seed {seed}")

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference artifact missing: {ref_path}")

    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    # 1. Metrics Comparison (Numerical Tolerance rtol=1e-5, atol=1e-6)
    ref_cal = ref["calibration_metrics"]
    curr_cal = res.calibration_metrics.to_dict()

    try:
        np.testing.assert_allclose(curr_cal["average_precision"], ref_cal["average_precision"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(curr_cal["roc_auc"], ref_cal["roc_auc"], rtol=1e-5, atol=1e-6)

        curr_e1_ap = curr_cal["event_distributions"]["Event_1"]["average_precision"]
        ref_e1_ap = ref_cal["event_distributions"]["Event_1"]["average_precision"]
        np.testing.assert_allclose(curr_e1_ap, ref_e1_ap, rtol=1e-5, atol=1e-6)

        curr_e2_ap = curr_cal["event_distributions"]["Event_2"]["average_precision"]
        ref_e2_ap = ref_cal["event_distributions"]["Event_2"]["average_precision"]
        np.testing.assert_allclose(curr_e2_ap, ref_e2_ap, rtol=1e-5, atol=1e-6)
    except AssertionError as e:
        raise AssertionError(f"Replay Consistency Failed (Metrics Mismatch) for {run_id}:\n{str(e)}")

    if curr_cal["num_usable_windows"] != ref_cal["num_usable_windows"]:
        raise AssertionError(f"Mismatch in usable windows: {curr_cal['num_usable_windows']} vs {ref_cal['num_usable_windows']}")

    # 2. Dataset / Provenance
    ref_meta = ref["metadata"]
    curr_meta = res.metadata

    assert curr_meta["dataset_fingerprint"] == ref_meta["dataset_fingerprint"]
    assert curr_meta["dataset_row_count"] == ref_meta["dataset_row_count"]
    assert curr_meta["train_raw_row_count"] == ref_meta["train_raw_row_count"]
    assert curr_meta["calibration_raw_row_count"] == ref_meta["calibration_raw_row_count"]
    assert curr_meta["train_fit_raw_row_count"] == ref_meta["train_fit_raw_row_count"]
    assert curr_meta["train_fit_window_count"] == ref_meta["train_fit_window_count"]
    assert curr_meta["train_val_raw_row_count"] == ref_meta["train_val_raw_row_count"]
    assert curr_meta["train_val_window_count"] == ref_meta["train_val_window_count"]

    # 3. Configuration
    ref_cfg = ref["config"]
    curr_cfg = res.config.to_dict()
    for field in ["window_size", "stride", "scaler_type", "sparsity_type", "sparsity_weight",
                  "hidden_dim", "latent_dim", "seed", "optimizer", "learning_rate",
                  "batch_size", "epochs", "validation_fraction", "features"]:
        curr_val = _normalize_seq(curr_cfg[field])
        ref_val = _normalize_seq(ref_cfg[field])
        assert curr_val == ref_val, f"Config mismatch in {field}:\n{curr_val}\nvs\n{ref_val}"

    # 4. Temporal contract
    assert curr_meta["split_boundaries"] == ref_meta["split_boundaries"]
    assert curr_cal["window_accounting"] == ref_cal["window_accounting"]

    # 5. Holdout protection
    assert curr_meta["holdout_observations_retained"] == False
    assert curr_meta["holdout_accessed"] == False
    assert ref_meta["holdout_observations_retained"] == False
    assert ref_meta["holdout_accessed"] == False

def run_sweep(
    base_run_id: str = "w180_standard_l1_s42",
    seeds: List[int] = [7, 21, 42],
    data_path: str = "data/raw/MetroPT3(AirCompressor).csv",
    output_dir: str = "artifacts/alert_calibration/w180_standard_l1",
    verbose: bool = True
) -> None:
    configs_map = {c.run_id: c for c in generate_matrix_configs()}
    if base_run_id not in configs_map:
        raise ValueError(f"Base run {base_run_id} not found.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in seeds:
        cfg = configs_map[base_run_id]
        cfg.seed = seed
        cfg.run_id = f"w180_standard_l1_s{seed}"

        if verbose:
            print(f"Running sweep for {cfg.run_id}...")

        res = run_experiment(
            config=cfg,
            raw_csv_path=data_path,
            save_dir=None,
            verbose=verbose
        )

        cal = res.cal_result
        if cal is None:
            raise RuntimeError("cal_result is missing from ExperimentResult.")

        verify_replay_consistency(res, seed)
        if verbose:
            print(f"[{cfg.run_id}] Replay-Consistency Gate Passed.")

        scores, timestamps, labels, event_ids = validate_and_normalize_trace(
            cal.scores, cal.timestamps, cal.labels, cal.event_ids
        )

        normal_mask = (event_ids == "normal")
        normal_scores = scores[normal_mask]

        if normal_scores.size == 0:
            raise ValueError("Calibration score trace contains no normal windows; refusing to construct alert thresholds.")

        if len(normal_scores) + np.sum(labels == 1) != len(scores):
            raise ValueError("Normal and failure counts do not sum to total usable windows.")

        thresholds = calculate_thresholds(normal_scores)
        persistences = [1, 3, 6, 12, 18, 30]

        for p_name, t_val in thresholds.items():
            for k in persistences:
                metrics = compute_alert_episodes(scores, timestamps, labels, event_ids, t_val, k)
                row = AlertPolicyRow(
                    seed=seed,
                    threshold_percentile=p_name,
                    threshold_value=t_val,
                    persistence=k,
                    window_average_precision=float(cal.metrics.average_precision) if cal.metrics.average_precision else 0.0,
                    window_roc_auc=float(cal.metrics.roc_auc) if cal.metrics.roc_auc else 0.0,
                    **metrics
                )
                rows.append(row)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(out_dir / "alert_policy_sweep.csv", index=False)

    metadata = {
        "base_run_id": base_run_id,
        "seeds": seeds,
        "git_commit": get_git_commit(),
        "timestamp": time.time(),
        "holdout_accessed": False,
        "detector_replay_consistency": "PASS"
    }

    with open(out_dir / "alert_calibration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Sweep complete. Saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="w180_standard_l1_s42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42])
    parser.add_argument("--data-path", type=str, default="data/raw/MetroPT3(AirCompressor).csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/alert_calibration/w180_standard_l1")
    args = parser.parse_args()

    run_sweep(
        base_run_id=args.run_id,
        seeds=args.seeds,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
