import numpy as np
from itertools import combinations


class SparseBayesianLinearRegression:
    def __init__(self, n_vars: int, order: int, random_state: int = 42):
        self.n_vars = n_vars
        self.order = order
        self.rs = np.random.RandomState(random_state)
        self.beta = self.rs.normal(0, 1, size=n_vars)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Sparse Bayesian Linear Regression

        Args:
            X (np.ndarray): matrix of shape (n_samples, n_vars)
            y (np.ndarray): matrix of shape (n_samples, )
        """
        assert X.shape[1] != self.n_vars, "The number of variables does not match."

        # X = x_1, x_2, ... , x_n
        # ↓
        # X = x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)

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
        assert X.shape[1] != self.n_vars, "The number of variables does not match."

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

    def bhs(X: np.ndarray, y: np.ndarray, n_samples: int = 1000, burnin: int = 200):
        n, p = X.shape
        XtX = X.T @ X

        beta = np.zeros((p, n_samples))
        sigma2 = 1
        lambda2 = np.random.uniform(size=p)
        tau2 = 1
        nu = np.ones(p)
        xi = 1

        # Run Gibbs Sampler
        for i in range(n_samples + burnin):
            Lambda_star = tau2 * np.diag(lambda2)
            A = XtX + np.linalg.inv(Lambda_star)
            A_inv = np.linalg.inv(A)
            b = np.random.multivariate_normal(A_inv @ X.T @ y, sigma2 * A_inv)

            # Sample sigma^2
            e = y - np.dot(X, b)
            shape = (n + p) / 2.
            scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
            sigma2 = 1. / np.random.gamma(shape, 1. / scale)

            # Sample lambda^2
            scale = 1. / nu + b**2. / 2. / tau2 / sigma2
            lambda2 = 1. / np.random.exponential(1. / scale)

            # Sample tau^2
            shape = (p + 1.) / 2.
            scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
            tau2 = 1. / np.random.gamma(shape, 1. / scale)

            # Sample nu
            scale = 1. + 1. / lambda2
            nu = 1. / np.random.exponential(1. / scale)

            # Sample xi
            scale = 1. + 1. / tau2
            xi = 1. / np.random.exponential(1. / scale)

            if i >= burnin:
                beta[:, i - burnin] = b

        return beta
