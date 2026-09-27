"""Data contract, validation, and causal window construction modules."""

from src.data.contract import (
    ALL_FEATURES,
    DIGITAL_SIGNALS,
    MAX_GAP_SECONDS,
    PRIMARY_FEATURES,
    SPLIT_BOUNDS,
    TIMESTAMP_COL,
    WINDOW_CANDIDATES,
    SplitPartition,
)
from src.data.validator import (
    ValidationReport,
    assign_split,
    detect_gaps,
    validate_dataset,
)
from src.data.windowing import CausalWindowBuilder

__all__ = [
    "ALL_FEATURES",
    "DIGITAL_SIGNALS",
    "MAX_GAP_SECONDS",
    "PRIMARY_FEATURES",
    "SPLIT_BOUNDS",
    "TIMESTAMP_COL",
    "WINDOW_CANDIDATES",
    "SplitPartition",
    "ValidationReport",
    "assign_split",
    "detect_gaps",
    "validate_dataset",
    "CausalWindowBuilder",
]
