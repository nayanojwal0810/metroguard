"""Selection and summary utilities for detector experiments."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

@dataclass
class MatrixSummaryRow:
    """Schema for the matrix summary artifact."""
    run_id: str
    w: int
    scaler: str
    sparsity: str
    sparsity_weight: float
    h: int
    z: int
    parameter_count: int
    average_precision: float
    pr_auc_trapezoidal: float
    roc_auc: float
    event_1_ap: Optional[float]
    event_2_ap: Optional[float]
    event_1_median: Optional[float]
    event_2_median: Optional[float]
    normal_median: float
    normal_p95: float
    normal_p99: float
    best_epoch: int
    best_val_loss: float
    runtime: float
    usable_calibration_windows: int
    exclusion_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RepeatabilityMetadata:
    """Metadata for the repeatability check."""
    run_id: str
    seeds: List[int]
    mean_ap: float
    std_ap: float
    mean_roc_auc: float
    worst_event_ap: float
    mean_runtime: float

def select_best_detector(summaries: List[MatrixSummaryRow]) -> MatrixSummaryRow:
    """Apply the predefined deterministic selection rule to choose the best configuration.
    
    Rule:
    Stage A: (Assumed already filtered for validity before passing to this function)
    Stage B: Rank descending by average_precision.
    Stage C: For near-ties (absolute AP diff <= 0.005 from the top), break ties by:
      1. Highest worst-event Average Precision
      2. Training stability (assumed checked in validity)
      3. Lower parameter count
    """
    if not summaries:
        raise ValueError("No summaries provided for selection.")

    # Sort primarily by AP descending
    sorted_summaries = sorted(summaries, key=lambda x: x.average_precision, reverse=True)
    top_ap = sorted_summaries[0].average_precision

    # Find all candidates within the practical near-tie band
    near_tie_band = [s for s in sorted_summaries if (top_ap - s.average_precision) <= 0.005]

    if len(near_tie_band) == 1:
        return near_tie_band[0]

    # Break ties
    def tie_breaker(s: MatrixSummaryRow):
        e1_ap = s.event_1_ap if s.event_1_ap is not None else 0.0
        e2_ap = s.event_2_ap if s.event_2_ap is not None else 0.0
        worst_event_ap = min(e1_ap, e2_ap)
        # Sort key: worst_event_ap (desc), parameter_count (asc)
        return (worst_event_ap, -s.parameter_count)

    # Sort the near-tie band by the tie-breaker
    best = sorted(near_tie_band, key=tie_breaker, reverse=True)[0]
    return best
