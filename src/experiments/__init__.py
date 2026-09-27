"""MetroGuard experiment harness package."""

from typing import Any
from src.experiments.config import ExperimentConfig, generate_matrix_configs
from src.experiments.evaluator import (
    CalibrationEvaluator,
    CalibrationMetrics,
    CalibrationResult,
)
from src.experiments.trainer import EpochMetrics, SAETrainer, TrainingHistory, set_seed

__all__ = [
    "CalibrationEvaluator",
    "CalibrationMetrics",
    "CalibrationResult",
    "EpochMetrics",
    "ExperimentConfig",
    "ExperimentResult",
    "SAETrainer",
    "TrainingHistory",
    "create_synthetic_experiment_data",
    "generate_matrix_configs",
    "run_experiment",
    "run_matrix",
    "run_smoke_test",
    "set_seed",
]


def __getattr__(name: str) -> Any:
    """Lazy import for runner components to prevent runpy execution warnings."""
    if name in (
        "ExperimentResult",
        "create_synthetic_experiment_data",
        "run_experiment",
        "run_matrix",
        "run_smoke_test",
    ):
        from src.experiments import runner

        return getattr(runner, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
