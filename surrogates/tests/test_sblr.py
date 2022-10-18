from typing import Callable
from surrogates._sblr import SparseBayesianLinearRegression
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal, assert_approx_equal


def _accuracy_callable(y_test: npt.NDArray, y_pred: npt.NDArray):
    return np.mean(y_test == y_pred)


def _mean_squared_error_callable(y_test: npt.NDArray, y_pred: npt.NDArray):
    return ((y_test - y_pred) ** 2).mean()


@pytest.fixture
def blp_dataset():
    def _blp_dataset(n_vars: int):
        _coefs = np.random.binomial(1, 0.1, size=n_vars)
        _coef0 = np.random.randn()

        X = np.random.randn(n_vars * 1000).reshape((-1, n_vars))
        y = X @ _coefs.T + _coef0
        coefs = np.append(_coef0, _coefs)
        return X, y, coefs

    return _blp_dataset


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_sblr(n_vars: int, blp_dataset: Callable):
    X, y, coefs = blp_dataset(n_vars)

    sblr = SparseBayesianLinearRegression(n_vars, order=1)
    sblr.fit(X, y)
    coefs_ = sblr.coefs

    assert n_vars + 1, sblr.n_coef
    assert_approx_equal(coefs_[0], coefs[0], significant=1)
    assert_array_almost_equal(coefs_[1:], coefs[1:], decimal=1)
