from typing import Callable
from surrogates._blr import BayesianLinearRegression
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose, assert_almost_equal


def _mean_squared_error_callable(y_test: npt.NDArray, y_pred: npt.NDArray):
    return ((y_test - y_pred) ** 2).mean()


def _root_mean_squared_error_callable(y_test: npt.NDArray, y_pred: npt.NDArray):
    return np.sqrt(_mean_squared_error_callable(y_test, y_pred))


def _sample_binary_matrix(n_samples: int, n_vars: int) -> npt.NDArray:
    # Generate matrix of zeros with ones along diagonals
    sample = np.zeros((n_samples, n_vars))

    # Sample model indices
    p = n_vars
    if n_vars > 64:
        p = 63
    sample_num = np.random.randint(2**p, size=n_samples)

    strformat = '{0:0' + str(n_vars) + 'b}'
    # Construct each binary model vector
    for i in range(n_samples):
        model = strformat.format(sample_num[i])
        sample[i, :] = np.array([int(b) for b in model])

    return sample


@pytest.fixture
def blp_dataset():
    def _blp_dataset(n_vars: int):
        mu = 2 * np.ones(n_vars)
        Sigma = np.eye(n_vars) / 20
        _coefs = np.random.multivariate_normal(mu, Sigma)
        _coef0 = np.random.randn()
        # noise
        eps = np.random.normal(0, 0.1)

        X = _sample_binary_matrix(1000, n_vars)
        y = X @ _coefs.T + _coef0 + eps
        coefs = np.append(_coef0, _coefs)
        return X, y, coefs, mu, Sigma

    return _blp_dataset


@pytest.fixture
def bqp_dataset():
    def _blp_dataset(n_vars: int):
        Q = np.random.randn(n_vars ** 2).reshape((n_vars, n_vars))
        Q = (Q + Q.T) / 2
        # noise
        eps = np.random.normal(0, 0.1)

        X = _sample_binary_matrix(1000, n_vars)
        y = np.diag(X @ Q @ X.T) + eps
        return X, y, Q

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
    assert_allclose(intercept_, intercept, atol=10e-1)
    assert_allclose(mu_, mu, atol=10e-1)
    assert_allclose(Sigma_, Sigma, atol=10e-1)


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_quadratic_blr(n_vars: int, bqp_dataset: Callable):
    X, y, Q = bqp_dataset(n_vars)
    intercept = np.mean(y)

    blr = BayesianLinearRegression(n_vars, order=2)
    blr.fit(X, y)
    intercept_ = blr.intercept_
    X_ = _sample_binary_matrix(1000, n_vars)
    y_ = blr.predict(X_)
    rmse = _root_mean_squared_error_callable(y_, X_ @ Q @ X_.T)

    assert int(n_vars * (n_vars - 1) / 2 + n_vars + 1) == blr.n_coef
    assert_almost_equal(intercept_, intercept, decimal=1)
    assert rmse < 10e-1
