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
        mu = 2
        Sigma = np.eye(n_vars) / 20
        _coefs = np.random.multivariate_normal(mu, np.eye(n_vars) / Sigma)
        _coef0 = np.random.randn()
        # noise
        eps = np.random.normal(0, 0.1)

        X = np.random.randn(n_vars * 1000).reshape((-1, n_vars))
        y = X @ _coefs.T + _coef0 + eps
        coefs = np.append(_coef0, _coefs)
        return X, y, coefs, mu, Sigma

    return _blp_dataset


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_blr(n_vars: int, blp_dataset: Callable):
    X, y, _, mu, Sigma = blp_dataset(n_vars)
    intercept = np.mean(y)

    blr = BayesianLinearRegression(n_vars, order=1)
    blr.fit(X, y)
    mu_ = blr.mu_
    Sigma_ = blr.Sigma_
    intercept_ = blr.intercept_

    assert n_vars + 1, blr.n_coef
    assert_approx_equal(intercept_, intercept, significant=1)
    assert_array_almost_equal(mu_, mu, decimal=1)
    assert_array_almost_equal(Sigma_, Sigma, decimal=1)
