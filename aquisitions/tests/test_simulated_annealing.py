from typing import Callable
from aquisitions import simulated_annealing
from itertools import product
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

np.random.seed(42)



@pytest.fixture
def blp_callable():
    def _blp_callable(n_vars: int):
        _coefs = np.random.randn(n_vars + 1)

        def objective(X: npt.NDArray):
            return _coefs[0] + _coefs[1:] @ X.T

        # Generate all cases
        X = np.array(list(map(list, product([0, 1], repeat=n_vars))))
        y = objective(X)

        # Find optimal solution
        max_idx = np.argmax(y)
        opt_x = X[max_idx, :]
        opt_y = y[max_idx]
        return objective, opt_x, opt_y

    return _blp_callable


@pytest.fixture
def bqp_callable():
    def _bqp_callable(n_vars: int):
        Q = np.random.randn(n_vars ** 2).reshape((n_vars, n_vars))

        def objective(X: npt.NDArray):
            return np.diag(X @ Q @ X.T)

        # Generate all cases
        X = np.array(list(map(list, product([0, 1], repeat=n_vars))))
        y = objective(X)

        # Find optimal solution
        max_idx = np.argmax(y)
        opt_x = X[max_idx, :]
        opt_y = y[max_idx]
        return objective, opt_x, opt_y

    return _bqp_callable


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_sa(n_vars: int, blp_callable: Callable):
    objective, opt_x, opt_y = blp_callable(n_vars)

    X, obj = simulated_annealing(objective, n_vars, n_iter=2 ** 16)
    max_idx = np.argmax(obj)
    x_ = X[max_idx, :]
    y_ = obj[max_idx]

    assert np.sum(x_ != opt_x) < int(n_vars / 5)
    assert_allclose(y_, opt_y, atol=10e0)


@pytest.mark.parametrize("n_vars", [5, 10])
def test_bqp_sa(n_vars: int, bqp_callable: Callable):
    objective, opt_x, opt_y = bqp_callable(n_vars)

    X, obj = simulated_annealing(objective, n_vars, n_iter=2 ** 16)
    max_idx = np.argmax(obj)
    x_ = X[max_idx, :]
    y_ = obj[max_idx]

    assert np.sum(x_ != opt_x) < int(n_vars / 5)
    assert_allclose(y_, opt_y, atol=10e0)
