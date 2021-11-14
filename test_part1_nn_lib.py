from typing import Tuple

import pytest
import numpy as np
from numpy.random import default_rng

from part1_nn_lib import Preprocessor, Trainer


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


@pytest.fixture
def trainer_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: A tuple of (input_dataset, target_dataset):
        - input_dataset: Array of input features, of shape
            (#_data_points, n_features) or (#_data_points,).
        - target_dataset: Array of corresponding targets, of
            shape (#_data_points, #output_neurons).
    """

    rg = default_rng(100)
    n_samples = 100
    weights = np.array([4, 2.5, 1.5])
    x = rg.random((n_samples, 2)) * 10.0
    x = np.hstack((x, np.ones((n_samples, 1))))
    y = np.matmul(x, weights)

    return x, y


def test_trainer_shuffle(trainer_data):
    """
    Tests 'Trainer.shuffle'.
    """

    input_dataset, target_dataset = trainer_data
    trainer = Trainer(None, len(input_dataset), 0, 0, "none", True)

    shuffled_input_dataset, shuffled_target_dataset = trainer.shuffle(
        input_dataset, target_dataset
    )

    # Test that shuffling changes the dataset order
    assert not np.all(
        input_dataset == shuffled_input_dataset
    ), "input dataset unchanged by shuffling"
    assert not np.all(
        target_dataset == shuffled_target_dataset
    ), "target dataset unchanged by shuffling"

    # Test that shuffling does not modify the data
    assert np.all(
        np.sort(input_dataset, axis=0) == np.sort(shuffled_input_dataset, axis=0)
    ), "input dataset modified by shuffling"
    assert np.all(
        np.sort(target_dataset, axis=0) == np.sort(shuffled_target_dataset, axis=0)
    ), "target dataset modified by shuffling"
