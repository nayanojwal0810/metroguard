"""Unit tests for leakage-safe scalers and strict fit/transform separation."""

import numpy as np
import pandas as pd
import pytest

from src.data.contract import PRIMARY_FEATURES
from src.preprocessing.scalers import MinMaxScaler, StandardScaler


def _create_mock_partition(mean_val: float, std_val: float, n: int = 100):
    """Generate synthetic sensor features with controlled mean and std."""
    np.random.seed(42)
    data = {}
    for feat in PRIMARY_FEATURES:
        data[feat] = np.random.normal(loc=mean_val, scale=std_val, size=n).astype(np.float32)
    return pd.DataFrame(data)


def test_standard_scaler_fit_transform_separation():
    """Verify that StandardScaler fits strictly on train and transforms holdout."""
    df_train = _create_mock_partition(mean_val=10.0, std_val=2.0, n=200)
    df_holdout = _create_mock_partition(mean_val=50.0, std_val=10.0, n=100)

    scaler = StandardScaler(features=PRIMARY_FEATURES)
    assert not scaler.is_fitted

    scaler.fit(df_train)
    assert scaler.is_fitted
    fitted_mean = scaler.mean_.copy()
    fitted_std = scaler.std_.copy()

    # Mean of training data must be close to 10.0
    np.testing.assert_allclose(fitted_mean, 10.0, atol=0.5)
    np.testing.assert_allclose(fitted_std, 2.0, atol=0.5)

    # Transform holdout
    transformed_holdout = scaler.transform(df_holdout)

    # CRITICAL INVARIANT: Transforming holdout must NOT mutate fitted training parameters
    np.testing.assert_array_equal(scaler.mean_, fitted_mean)
    np.testing.assert_array_equal(scaler.std_, fitted_std)

    # Check that holdout values were scaled by train parameters:
    # holdout values around 50 should have z-score approx (50 - 10) / 2 = 20
    sample_feat = PRIMARY_FEATURES[0]
    expected_z = (df_holdout[sample_feat].to_numpy() - fitted_mean[0]) / fitted_std[0]
    np.testing.assert_allclose(transformed_holdout[sample_feat].to_numpy(), expected_z, atol=1e-5)


def test_minmax_scaler_fit_transform_separation():
    """Verify that MinMaxScaler fits strictly on train and transforms holdout."""
    df_train = pd.DataFrame(
        {
            "TP2": [0.0, 5.0, 10.0],
            "TP3": [10.0, 20.0, 30.0],
            "H1": [0.0, 1.0, 2.0],
            "DV_pressure": [0.0, 0.5, 1.0],
            "Reservoirs": [5.0, 7.5, 10.0],
            "Oil_temperature": [50.0, 60.0, 70.0],
            "Motor_current": [0.0, 5.0, 10.0],
        }
    )
    # Holdout has an extreme anomalous leak state exceeding training bounds
    df_holdout = pd.DataFrame(
        {
            "TP2": [15.0],  # Above max
            "TP3": [-5.0],  # Below min
            "H1": [1.0],
            "DV_pressure": [0.5],
            "Reservoirs": [2.0],  # Below min
            "Oil_temperature": [85.0],  # Above max
            "Motor_current": [20.0],  # Above max
        }
    )

    scaler = MinMaxScaler(features=PRIMARY_FEATURES)
    scaler.fit(df_train)

    train_transformed = scaler.transform(df_train)
    # Train transformed values must be within [0, 1]
    assert train_transformed["TP2"].min() == pytest.approx(0.0)
    assert train_transformed["TP2"].max() == pytest.approx(1.0)

    # Holdout transformed using train parameters can and should reflect out-of-bounds anomalies
    holdout_transformed = scaler.transform(df_holdout)
    assert holdout_transformed["TP2"].iloc[0] == pytest.approx(1.5)  # (15 - 0) / 10 = 1.5
    assert holdout_transformed["TP3"].iloc[0] == pytest.approx(-0.75)  # (-5 - 10) / 20 = -0.75


def test_scaler_raises_if_transform_before_fit():
    """Verify runtime error when attempting to transform unfitted data."""
    df = _create_mock_partition(mean_val=5.0, std_val=1.0, n=10)
    scaler = StandardScaler(features=PRIMARY_FEATURES)

    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(df)

    minmax = MinMaxScaler(features=PRIMARY_FEATURES)
    with pytest.raises(RuntimeError, match="must be fitted"):
        minmax.transform(df)
