from typing import Callable
from surrogates import SGDRegressor
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
    sample_num = np.random.randint(2**n_vars, size=n_samples)

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
        coefs = np.random.multivariate_normal(mu, Sigma)
        coef0 = np.random.randn()

        X = _sample_binary_matrix(1000, n_vars)
        y = X @ coefs.T + coef0
        return X, y, coefs, coef0

    return _blp_dataset


@pytest.fixture
def bqp_dataset():
    def _bqp_dataset(n_vars: int):
        # Sparse matrix
        Q = np.random.randn(n_vars ** 2).reshape((n_vars, n_vars))

        X = _sample_binary_matrix(1000, n_vars)
        y = np.diag(X @ Q @ X.T)
        return X, y, Q

    return _bqp_dataset


@pytest.mark.parametrize("n_vars", [5, 10, 15])
def test_linear_blr(n_vars: int, blp_dataset: Callable):
    X, y, coefs, _ = blp_dataset(n_vars)
    intercept = np.mean(y)

    model = SGDRegressor(n_vars, order=1)
    model.fit(X, y, batch_size=8)

    intercept_ = model.intercept_
    X_ = _sample_binary_matrix(1000, n_vars)
    y_ = model.predict(X_)
    rmse = _root_mean_squared_error_callable(y_, coefs @ X_.T)

    assert_equal(model.n_coef_, n_vars)
    assert_allclose(intercept_, intercept, atol=10e-1)
    assert rmse < 10e1


@pytest.mark.parametrize("n_vars", [5, 10])
def test_quadratic_blr(n_vars: int, bqp_dataset: Callable):
    X, y, Q = bqp_dataset(n_vars)
    intercept = np.mean(y)

    model = SGDRegressor(n_vars, order=2)
    model.fit(X, y, batch_size=8)

    intercept_ = model.intercept_
    X_ = _sample_binary_matrix(1000, n_vars)
    y_ = model.predict(X_)
    rmse = _root_mean_squared_error_callable(y_, X_ @ Q @ X_.T)

    assert_equal(model.n_coef_, int(n_vars * (n_vars - 1) / 2 + n_vars))
    assert_allclose(intercept_, intercept, atol=10e-1)
    assert rmse < 10e1
