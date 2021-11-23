from typing import Tuple
from itertools import chain

import pytest
import numpy as np
import pandas as pd

from part2_house_value_regression import Regressor


@pytest.fixture
def housing_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :return: A tuple of (x, y):
        - x: Pandas Dataframe of shape (n, k) and mixed type where n is the
            number of instances and k is the number of features.
        - y: Pandas Dataframe of shape (n, 1) and type float where n is the
            number of instances.
    """

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    return x, y


def test_regressor_preprocessor(housing_data):
    """
    Tests '_preprocessor' method of 'Regressor' class.
    """
    standardization_bool = True

    x, y = housing_data
    regressor = Regressor(x)

    # Training mode
    x_norm_train, y_norm_train = regressor._preprocessor(
        x, y, training=True, standardization=standardization_bool
    )
    # Default mode (using parameters from training mode)
    x_norm, y_norm = regressor._preprocessor(x, y, standardization=standardization_bool)
    # Default mode without y
    x_norm2, _ = regressor._preprocessor(x, standardization=standardization_bool)

    # Group for cleaner testing
    X = [x_norm_train, x_norm, x_norm2]
    Y = [y_norm_train, y_norm]

    # Test type
    for array in chain(X, Y):
        assert isinstance(array, np.ndarray), f"type {type(array)} is not np.ndarray"

    # Test shape
    for x_norm in X:
        assert (
            x_norm.shape[0] == x.shape[0] and x_norm.shape[1] >= x.shape[1]
        ), f"shapes {x_norm.shape} and {x.shape} don't match"
    for y_norm in Y:
        assert (
            y_norm.shape == y.shape
        ), f"shapes {y_norm.shape} and {y.shape} are not equal"

    # Test if arrays contain np.nan or np.inf
    for array in chain(X, Y):
        assert np.isfinite(array).all()

    # Test equivalence
    assert np.allclose(x_norm_train, x_norm) and np.allclose(x_norm, x_norm2)
    assert np.array_equal(y_norm_train, y_norm)

    # Test normalisation
    for x_norm in X:

        # Uncommented this line and comment the other line when using min-max
        if standardization_bool:
            # Test Standardization
            assert np.all(np.mean(x_norm, axis=0) == pytest.approx(0)) and np.all(
                np.var(x_norm, axis=0) == pytest.approx(1)
            )
        else:
            # Test MinMax normalization
            assert np.all(x_norm <= 1) and np.all(x_norm >= 0)
