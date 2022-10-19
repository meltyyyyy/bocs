from typing import Callable
from surrogates._blr import BayesianLinearRegression
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


@pytest.fixture
def blp_dataset():
    def _blp_dataset(n_vars: int):
        mu = 2 * np.ones(n_vars)
        Sigma = np.eye(n_vars) / 20
        _coefs = np.random.multivariate_normal(mu, Sigma)
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

    assert n_vars + 1 == blr.n_coef
    assert_almost_equal(intercept_, intercept, decimal=1)
    assert_allclose(mu_, mu, atol=10e-1)
    assert_allclose(Sigma_, Sigma, atol=10e-1)
