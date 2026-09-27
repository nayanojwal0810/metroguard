"""Data contract definitions, schema specifications, and split boundaries."""

from dataclasses import dataclass
from enum import Enum
from typing import Final, Tuple
import pandas as pd

# Canonical feature subsets
PRIMARY_FEATURES: Final[Tuple[str, ...]] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)

DIGITAL_SIGNALS: Final[Tuple[str, ...]] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

ALL_FEATURES: Final[Tuple[str, ...]] = PRIMARY_FEATURES + DIGITAL_SIGNALS

TIMESTAMP_COL: Final[str] = "timestamp"
INDEX_COL: Final[str] = "column00"

# Operational engineering rules
MAX_GAP_SECONDS: Final[float] = 60.0
WINDOW_CANDIDATES: Final[Tuple[int, ...]] = (6, 30, 90, 180)

# Dataset identity
DATASET_CANONICAL_NAME: Final[str] = "MetroPT3(AirCompressor).csv"
DATASET_EXPECTED_ROWS: Final[int] = 1_516_948
DATASET_SHA256: Final[str] = (
    "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"
)


class SplitPartition(str, Enum):
    """Chronological partitions for evaluation."""

    TRAIN = "TRAIN"
    CALIBRATION = "CALIBRATION"
    HOLDOUT = "HOLDOUT"
    UNUSED_TAIL = "UNUSED_TAIL"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


@dataclass(frozen=True)
class PartitionBoundary:
    """Inclusive start and end timestamps for a partition."""

    name: SplitPartition
    start: pd.Timestamp
    end: pd.Timestamp


SPLIT_BOUNDS: Final[Tuple[PartitionBoundary, ...]] = (
    PartitionBoundary(
        name=SplitPartition.TRAIN,
        start=pd.Timestamp("2020-02-01 00:00:00"),
        end=pd.Timestamp("2020-03-31 23:59:59"),
    ),
    PartitionBoundary(
        name=SplitPartition.CALIBRATION,
        start=pd.Timestamp("2020-04-01 00:00:00"),
        end=pd.Timestamp("2020-05-31 23:59:59"),
    ),
    PartitionBoundary(
        name=SplitPartition.HOLDOUT,
        start=pd.Timestamp("2020-06-01 00:00:00"),
        end=pd.Timestamp("2020-08-31 23:59:59"),
    ),
    PartitionBoundary(
        name=SplitPartition.UNUSED_TAIL,
        start=pd.Timestamp("2020-09-01 00:00:00"),
        end=pd.Timestamp("2020-09-01 03:59:50"),
    ),
)


@dataclass(frozen=True)
class FailureEvent:
    """Documented maintenance failure event interval."""

    event_id: str
    partition: SplitPartition
    start: pd.Timestamp
    end: pd.Timestamp
    description: str


FAILURE_EVENTS: Final[Tuple[FailureEvent, ...]] = (
    FailureEvent(
        event_id="Event_1",
        partition=SplitPartition.CALIBRATION,
        start=pd.Timestamp("2020-04-18 00:00:00"),
        end=pd.Timestamp("2020-04-18 23:59:59"),
        description="Air leak on clients (pipe blowout; severe pressure drop)",
    ),
    FailureEvent(
        event_id="Event_2",
        partition=SplitPartition.CALIBRATION,
        start=pd.Timestamp("2020-05-29 23:30:00"),
        end=pd.Timestamp("2020-05-30 06:00:00"),
        description="Air leak on air dryer (pilot valve malfunction; LPS triggers)",
    ),
    FailureEvent(
        event_id="Event_3",
        partition=SplitPartition.HOLDOUT,
        start=pd.Timestamp("2020-06-05 10:00:00"),
        end=pd.Timestamp("2020-06-07 14:30:00"),
        description="Air leak (sustained pressure drops; maintenance 8-Jun)",
    ),
    FailureEvent(
        event_id="Event_4",
        partition=SplitPartition.HOLDOUT,
        start=pd.Timestamp("2020-07-15 14:30:00"),
        end=pd.Timestamp("2020-07-15 19:00:00"),
        description="Air leak (maintenance 16-Jul 00:00)",
    ),
)
