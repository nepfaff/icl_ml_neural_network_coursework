import pytest
import numpy as np
from numpy.random import default_rng

from part1_nn_lib import Preprocessor


@pytest.fixture
def preprocessor_data() -> np.ndarray:
    """
    :return: A list of arrays of input features, of shape
        (#_data_points, n_features) or (#_data_points,).
    """

    rg = default_rng(100)

    X = [rg.random((100, 2)) * 10.0, np.arange(-5, 871, 1.4)]

    return X


def test_preprocessor(preprocessor_data):
    """
    Tests 'Preprocessor'.
    """

    X = preprocessor_data

    for i, x in enumerate(X):
        preprocessor = Preprocessor(x)
        normalized_x = preprocessor.apply(x)
        original_x = preprocessor.revert(normalized_x)

        # Test apply method
        assert np.all(normalized_x <= 1) and np.all(
            normalized_x >= 0
        ), f"test case {i} failed: normalized values are not between zero and one"

        # Test revert method
        assert np.allclose(
            x, original_x
        ), f"test case {i} failed: revert does not produce the original dataset"
