from utils import sample_binary_matrix
import pytest
import numpy as np


@pytest.mark.parametrize("n_vars", [1, 10, 100])
def test_sample_binary_matrix(n_vars: int):
    n_samples = 100
    X = sample_binary_matrix(n_samples, n_vars)

    assert X.shape[0], n_samples
    assert X.shape[1], n_vars
    assert np.all((X == 0) | (X == 1))
