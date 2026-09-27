"""Tests for detector selection and matrix summary schema."""

import pytest
from src.experiments.selection import MatrixSummaryRow, RepeatabilityMetadata, select_best_detector

def test_matrix_summary_schema():
    row = MatrixSummaryRow(
        run_id="test_run",
        w=30,
        scaler="StandardScaler",
        sparsity="l1",
        sparsity_weight=1e-4,
        h=128,
        z=32,
        parameter_count=50000,
        average_precision=0.5,
        pr_auc_trapezoidal=0.4,
        roc_auc=0.9,
        event_1_ap=0.3,
        event_2_ap=0.6,
        event_1_median=1.0,
        event_2_median=2.0,
        normal_median=0.1,
        normal_p95=0.5,
        normal_p99=0.8,
        best_epoch=10,
        best_val_loss=0.01,
        runtime=150.0,
        usable_calibration_windows=400000,
        exclusion_counts={"gap": 0}
    )
    d = row.to_dict()
    assert d["run_id"] == "test_run"
    assert d["w"] == 30
    assert d["average_precision"] == 0.5

def test_repeatability_metadata():
    rm = RepeatabilityMetadata(
        run_id="test_run",
        seeds=[7, 21, 42],
        mean_ap=0.5,
        std_ap=0.01,
        mean_roc_auc=0.9,
        worst_event_ap=0.3,
        mean_runtime=150.0
    )
    assert len(rm.seeds) == 3

def test_select_best_detector_simple():
    s1 = MatrixSummaryRow(run_id="run1", w=30, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=100, 
                          average_precision=0.6, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.6, event_2_ap=0.6,
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    s2 = MatrixSummaryRow(run_id="run2", w=30, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=100, 
                          average_precision=0.4, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.4, event_2_ap=0.4,
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    best = select_best_detector([s1, s2])
    assert best.run_id == "run1"

def test_select_best_detector_near_tie_worst_event():
    # AP diff is 0.003 (within 0.005)
    s1 = MatrixSummaryRow(run_id="run1", w=30, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=100, 
                          average_precision=0.600, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.2, event_2_ap=0.6, # worst is 0.2
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    s2 = MatrixSummaryRow(run_id="run2", w=30, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=100, 
                          average_precision=0.597, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.4, event_2_ap=0.4, # worst is 0.4
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    
    # s2 should win due to better worst-event AP
    best = select_best_detector([s1, s2])
    assert best.run_id == "run2"

def test_select_best_detector_near_tie_param_count():
    # AP diff is 0.003, worst event AP is tied at 0.4
    s1 = MatrixSummaryRow(run_id="run1", w=90, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=500, 
                          average_precision=0.600, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.4, event_2_ap=0.6,
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    s2 = MatrixSummaryRow(run_id="run2", w=30, scaler="std", sparsity="l1", sparsity_weight=0, h=0, z=0, parameter_count=100, 
                          average_precision=0.598, pr_auc_trapezoidal=0, roc_auc=0, event_1_ap=0.6, event_2_ap=0.4,
                          event_1_median=0, event_2_median=0, normal_median=0, normal_p95=0, normal_p99=0, best_epoch=0, best_val_loss=0, runtime=0, usable_calibration_windows=0, exclusion_counts={})
    
    # s2 should win due to lower parameter count
    best = select_best_detector([s1, s2])
    assert best.run_id == "run2"
