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
    total_alerts: int
    normal_alerts: int
    false_alerts_per_30_days: float
    event_1_detected: bool
    event_1_lead_time_sec: float
    event_2_detected: bool
    event_2_lead_time_sec: float

def calculate_thresholds(normal_scores: np.ndarray) -> Dict[float, float]:
    percentiles = [95.0, 97.0, 98.0, 99.0, 99.5, 99.7, 99.9, 99.95, 99.99]
    return {p: float(np.percentile(normal_scores, p)) for p in percentiles}

def compute_alert_episodes(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    labels: np.ndarray,
    event_ids: np.ndarray,
    threshold: float,
    persistence: int
) -> Dict[str, Any]:
    n = len(scores)
    above_threshold = (scores >= threshold)
    
    # Identify gaps > 60s
    time_diffs = np.diff(timestamps.values) / np.timedelta64(1, 's')
    is_gap = np.insert(time_diffs > 60.0, 0, False)
    
    alerts = []
    current_streak = 0
    in_alert_episode = False
    
    for i in range(n):
        if is_gap[i]:
            current_streak = 0
            in_alert_episode = False
            
        if above_threshold[i]:
            current_streak += 1
            if current_streak >= persistence and not in_alert_episode:
                alerts.append(i - persistence + 1)
                in_alert_episode = True
        else:
            current_streak = 0
            in_alert_episode = False

    total_duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
    if total_duration_sec == 0:
        total_duration_sec = 1
        
    days = total_duration_sec / 86400.0
    
    normal_alerts = 0
    e1_det = False
    e1_lt = 0.0
    e2_det = False
    e2_lt = 0.0
    
    e1_start = pd.Timestamp("2020-04-18 00:00:00")
    e2_start = pd.Timestamp("2020-05-29 23:30:00")
    
    for idx in alerts:
        t = timestamps[idx]
        eid = event_ids[idx]
        if eid == "None":
            normal_alerts += 1
        elif eid == "Event_1":
            if not e1_det:
                e1_det = True
                e1_lt = (e1_start - t).total_seconds()
                if e1_lt < 0:
                    e1_lt = 0.0 # post-event detection is not early warning
        elif eid == "Event_2":
            if not e2_det:
                e2_det = True
                e2_lt = (e2_start - t).total_seconds()
                if e2_lt < 0:
                    e2_lt = 0.0

    return {
        "total_alerts": len(alerts),
        "normal_alerts": normal_alerts,
        "false_alerts_per_30_days": (normal_alerts / days) * 30 if days > 0 else 0,
        "event_1_detected": e1_det,
        "event_1_lead_time_sec": e1_lt,
        "event_2_detected": e2_det,
        "event_2_lead_time_sec": e2_lt
    }

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
            
        scores = cal.scores
        timestamps = cal.timestamps
        labels = cal.labels
        event_ids = cal.event_ids
        
        normal_mask = (event_ids == "None")
        normal_scores = scores[normal_mask]
        
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
        "holdout_accessed": False
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
