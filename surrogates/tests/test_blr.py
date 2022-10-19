from typing import Callable
from surrogates._blr import BayesianLinearRegression
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
        _coefs = np.random.normal(0, 0.1, size=n_vars)
        _coef0 = np.random.randn()
        # noise
        eps = np.random.normal(0, 0.1)

        X = np.random.randn(n_vars * 1000).reshape((-1, n_vars))
        y = X @ _coefs.T + _coef0 + eps
        coefs = np.append(_coef0, _coefs)
        return X, y, coefs, eps

    return _blp_dataset


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_blr(n_vars: int, blp_dataset: Callable):
    X, y, coefs = blp_dataset(n_vars)

    # since blp dataset use alpha=0.1,  sigma=0.1
    blr = BayesianLinearRegression(n_vars, order=1, alpha=0.1, sigma=0.1)
    blr.fit(X, y)
    

    assert n_vars + 1, blr.n_coef
    assert_approx_equal(coefs_[0], coefs[0], significant=1)
    assert_array_almost_equal(coefs_[1:], coefs[1:], decimal=1)
