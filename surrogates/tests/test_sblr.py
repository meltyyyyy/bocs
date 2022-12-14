from typing import Callable
from surrogates._sblr import SparseBayesianLinearRegressor
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose, assert_equal


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
        _coefs = np.random.binomial(1, 0.1, size=n_vars)
        _coef0 = np.random.randn()

        X = _sample_binary_matrix(1000, n_vars)
        y = X @ _coefs.T + _coef0
        coefs = np.append(_coef0, _coefs)
        return X, y, coefs

    return _blp_dataset


@pytest.fixture
def bqp_dataset():
    def _bqp_dataset(n_vars: int):
        # Sparse matrix
        Q = np.random.randn(n_vars ** 2).reshape((n_vars, n_vars))
        i = np.linspace(1, n_vars, n_vars)
        j = np.linspace(1, n_vars, n_vars)
        def K(s, t): return np.exp(-1 * (s - t)**2)
        decay = K(i[:, None], j[None, :])
        Q = (Q + Q.T) / 2
        Q = Q * decay
        Q[Q < 10e-4] = 0

        # noise
        eps = np.random.normal(0, 0.1)

        X = _sample_binary_matrix(1000, n_vars)
        y = np.diag(X @ Q @ X.T) + eps
        return X, y, Q

    return _bqp_dataset


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_sblr(n_vars: int, blp_dataset: Callable):
    X, y, coefs = blp_dataset(n_vars)

    sblr = SparseBayesianLinearRegressor(n_vars, order=1)
    sblr.fit(X, y)
    coefs_ = sblr.coefs

    assert_equal(sblr.n_coef, n_vars + 1)
    assert_allclose(coefs_[0], coefs[0], atol=10e0)
    assert_allclose(coefs_[1:], coefs[1:], atol=10e0)


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_quadratic_sblr(n_vars: int, bqp_dataset: Callable):
    X, y, Q = bqp_dataset(n_vars)
    intercept = np.mean(y)

    sblr = SparseBayesianLinearRegressor(n_vars, order=2)
    sblr.fit(X, y)
    intercept_ = sblr.coefs[0]
    X_ = _sample_binary_matrix(1000, n_vars)
    y_ = sblr.predict(X_)
    rmse = _root_mean_squared_error_callable(y_, X_ @ Q @ X_.T)

    assert_equal(sblr.n_coef, int(n_vars * (n_vars - 1) / 2 + n_vars + 1))
    assert_allclose(intercept_, intercept, atol=10e-1)
    assert rmse < 10e1
