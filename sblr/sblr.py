from typing import Union
import numpy as np
from itertools import combinations

rs = np.random.RandomState(42)


class SparseBayesianLinearRegression:
    def __init__(self, n_vars: int, order: int, random_state: int = 42):
        assert n_vars > 0, "The number of variables must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.rs = np.random.RandomState(random_state)
        self.n_coef = int(1 + n_vars + 0.5 * n_vars * (n_vars - 1))
        self.coefs = self.rs.normal(0, 1, size=self.n_coef)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Sparse Bayesian Linear Regression

        Args:
            X (np.ndarray): matrix of shape (n_samples, n_vars)
            y (np.ndarray): matrix of shape (n_samples, )
        """
        assert X.shape[1] == self.n_vars, "The number of variables does not match. X has {} variables, but n_vars is {}.".format(
            X.shape[1], self.n_vars)
        assert y.ndim == 1, "y should be 1 dimension of shape (n_samples, ), but is {}".format(
            y.ndim)

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)

        needs_sample = 1
        while (needs_sample):
            try:
                _coefs, _coef0 = self._bhs(X, y)
            except Exception as e:
                print(e)
                continue

            if not np.isnan(_coefs).any():
                needs_sample = 0

        self.coefs = np.append(_coef0, _coefs)

    def predict(self, x: np.ndarray) -> np.float64:
        assert x.shape[1] == self.n_vars, "The number of variables does not match. X has {} variables, but n_vars is {}.".format(
            X.shape[1], self.n_vars)

        x = self._order_effects(x)
        x = np.append(1, x)
        return x @ self.coefs

    def _order_effects(self, X: np.ndarray) -> np.ndarray:
        """Compute order effects
        Computes data matrix for all coupling
        orders to be added into linear regression model.

        Order is the number of combinations that needs to be taken into consideration,
        usually set to 2.

        Args:
            X (np.ndarray): input materix of shape (n_samples, n_vars)

        Returns:
            X_allpairs (np.ndarray): all combinations of variables up to consider,
                                     which shape is (n_samples, Σ[i=1, order] comb(n_vars, i))
        """
        assert X.shape[1] == self.n_vars, "The number of variables does not match. X has {} variables, but n_vars is {}.".format(
            X.shape[1], self.n_vars)

        n_samples, n_vars = X.shape
        X_allpairs = X.copy()

        for i in range(2, self.order + 1, 1):

            # generate all combinations of indices (without diagonals)
            offdProd = np.array(list(combinations(np.arange(n_vars), i)))

            # generate products of input variables
            x_comb = np.zeros((n_samples, offdProd.shape[0], i))
            for j in range(i):
                x_comb[:, :, j] = X[:, offdProd[:, j]]
            X_allpairs = np.append(X_allpairs, np.prod(x_comb, axis=2), axis=1)

        return X_allpairs

    def _bhs(self, X: np.ndarray, y: np.ndarray, n_samples: int = 1,
             burnin: int = 50) -> Union[np.ndarray, np.float64]:
        assert X.shape[1] == self.n_coef - 1, "The number of combinations is wrong, it should be {}".format(
            self.n_coef)
        assert y.ndim == 1, "y should be 1 dimension of shape (n_samples, ), but is {}".format(
            y.ndim)

        n, p = X.shape
        XtX = X.T @ X

        beta = np.zeros((p, n_samples))
        beta0 = np.mean(y)
        sigma2 = 1
        lambda2 = self.rs.uniform(size=p)
        tau2 = 1
        nu = np.ones(p)
        xi = 1

        # Run Gibbs Sampler
        for i in range(n_samples + burnin):
            Lambda_star = tau2 * np.diag(lambda2)
            A = XtX + np.linalg.inv(Lambda_star)
            A_inv = np.linalg.inv(A)
            b = self.rs.multivariate_normal(A_inv @ X.T @ y, sigma2 * A_inv)

            # Sample sigma^2
            e = y - np.dot(X, b)
            shape = (n + p) / 2.
            scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
            sigma2 = 1. / self.rs.gamma(shape, 1. / scale)

            # Sample lambda^2
            scale = 1. / nu + b**2. / 2. / tau2 / sigma2
            lambda2 = 1. / self.rs.exponential(1. / scale)

            # Sample tau^2
            shape = (p + 1.) / 2.
            scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
            tau2 = 1. / self.rs.gamma(shape, 1. / scale)

            # Sample nu
            scale = 1. + 1. / lambda2
            nu = 1. / self.rs.exponential(1. / scale)

            # Sample xi
            scale = 1. + 1. / tau2
            xi = 1. / self.rs.exponential(1. / scale)

            if i >= burnin:
                beta[:, i - burnin] = b

        return beta, beta0


def sample_X(n_samples: int, n_vars: int) -> np.ndarray:
    """Sample binary matrix

    Args:
        n_samples (int): The number of samples.
        n_vars (int): The number of variables.

    Returns:
        np.ndarray: Binary matrix of shape (n_samples, n_vars)
    """
    # Generate matrix of zeros with ones along diagonals
    sample = np.zeros((n_samples, n_vars))

    # Sample model indices
    sample_num = rs.randint(2**n_vars, size=n_samples)

    strformat = '{0:0' + str(n_vars) + 'b}'
    # Construct each binary model vector
    for i in range(n_samples):
        model = strformat.format(sample_num[i])
        sample[i, :] = np.array([int(b) for b in model])

    return sample


if __name__ == '__main__':
    n_vars = 10
    n_samples = 10
    # Q = rs.randn(n_vars**2).reshape(n_vars, n_vars)
    Q = np.eye(n_vars)

    def objective(X: np.ndarray) -> np.float64:
        return np.diag(X @ Q @ X.T)

    X = sample_X(n_samples, n_vars)
    y = objective(X)

    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)
