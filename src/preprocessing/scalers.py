"""Leakage-safe scalers enforcing parameter fitting strictly on training partitions."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.data.contract import PRIMARY_FEATURES


class BaseScaler(ABC):
    """Abstract base class for leakage-safe feature scalers."""

    def __init__(self, features: Sequence[str] = PRIMARY_FEATURES) -> None:
        self.features = tuple(features)
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> "BaseScaler":
        """Compute scaling parameters strictly from the provided training data."""
        pass

    @abstractmethod
    def transform(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply scaling transformation using previously fitted parameters."""
        pass

    def fit_transform(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit on training data and return transformed representation."""
        return self.fit(data).transform(data)


class StandardScaler(BaseScaler):
    """Standardizes features by removing the mean and scaling to unit variance.

    Parameters are fitted strictly on training data:
        z = (x - mu_train) / sigma_train
    """

    def __init__(
        self,
        features: Sequence[str] = PRIMARY_FEATURES,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(features=features)
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> "StandardScaler":
        """Compute mean and standard deviation strictly on training data."""
        if isinstance(data, pd.DataFrame):
            x = data[list(self.features)].to_numpy(dtype=np.float64)
        else:
            x = np.asarray(data, dtype=np.float64)

        if len(x) == 0:
            raise ValueError("Cannot fit scaler on empty data.")

        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ < self.eps] = 1.0
        self.is_fitted = True
        return self

    def transform(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using training mean and standard deviation."""
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted on training data before transforming.")

        if isinstance(data, pd.DataFrame):
            x = data[list(self.features)].to_numpy(dtype=np.float64)
            scaled = (x - self.mean_) / self.std_
            result = data.copy()
            for i, feat in enumerate(self.features):
                result[feat] = scaled[:, i].astype(np.float32)
            return result
        else:
            x = np.asarray(data, dtype=np.float64)
            scaled = (x - self.mean_) / self.std_
            return scaled.astype(np.float32)


class MinMaxScaler(BaseScaler):
    """Transforms features by scaling each feature to the [0, 1] range based on training bounds.

    Parameters are fitted strictly on training data:
        x_scaled = (x - min_train) / (max_train - min_train)
    """

    def __init__(
        self,
        features: Sequence[str] = PRIMARY_FEATURES,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(features=features)
        self.eps = eps
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> "MinMaxScaler":
        """Compute min and max bounds strictly on training data."""
        if isinstance(data, pd.DataFrame):
            x = data[list(self.features)].to_numpy(dtype=np.float64)
        else:
            x = np.asarray(data, dtype=np.float64)

        if len(x) == 0:
            raise ValueError("Cannot fit scaler on empty data.")

        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        self.range_ = self.max_ - self.min_
        # Prevent division by zero for constant features
        self.range_[self.range_ < self.eps] = 1.0
        self.is_fitted = True
        return self

    def transform(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using training min and max bounds."""
        if not self.is_fitted or self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler must be fitted on training data before transforming.")

        if isinstance(data, pd.DataFrame):
            x = data[list(self.features)].to_numpy(dtype=np.float64)
            scaled = (x - self.min_) / self.range_
            result = data.copy()
            for i, feat in enumerate(self.features):
                result[feat] = scaled[:, i].astype(np.float32)
            return result
        else:
            x = np.asarray(data, dtype=np.float64)
            scaled = (x - self.min_) / self.range_
            return scaled.astype(np.float32)
