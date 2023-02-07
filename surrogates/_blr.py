import numpy as np
import numpy.typing as npt
from scipy.special import comb
from dotenv import load_dotenv
from utils import make_qubo, order_effects

load_dotenv()


class BayesianLinearRegressor:
    def __init__(self, n_vars: int, order: int, alpha: float = 1e-1, beta: float = 1e-1, random_state: int = 42):
        """
        Bayesian Linear Regression.

        Args:
            n_vars (int): Number of variables
            order (int): Statisical model order
            alpha (int): Precision parameter for zero-mean isotropic Gaussian error
            beta (int): Precision parameter for zero-mean isotropic Gaussian priror
        """
        assert n_vars > 0, "The number of variables must be greater than 0"
        assert order > 0, "order must be greater than 0"
        assert alpha > 0, "alpha must be greater than 0"
        assert beta > 0, "sigma must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.alpha_ = alpha
        self.beta_ = beta
        self.rs = np.random.RandomState(random_state)
        self.n_coef_ = int(np.sum([comb(n_vars, i) for i in range(order + 1)]))
        self.coef_ = np.random.rand(self.n_coef_)
        self.intercept_ = 0
        self.mu_ = None
        self.Sigma_ = None

    def fit(self, X: npt.NDArray, y: npt.NDArray):
        """
        Fit Bayesian Linear Regression.

        Args:
            X (npt.NDArray): Matrix of shape (n_samples, n_vars)
            y (npt.NDArray): Matrix of shape (n_samples, )
        """
        assert X.shape[1] == self.n_vars,\
            "The number of variables does not match. \
            X has {} variables, but n_vars is {}.".format(X.shape[1], self.n_vars)
        assert y.ndim == 1, \
            "y should be 1 dimension of shape (n_samples, ), but is {}".format(
                y.ndim)

        alpha = self.alpha_
        beta = self.beta_

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)

        XtX = X.T @ X

        # compute covariace
        inner_term = alpha * XtX + beta * np.eye(X.shape[1])
        Sigma = np.linalg.inv(inner_term)

        # compute mean
        mu = alpha * np.linalg.inv(inner_term) @ X.T @ y

        self.intercept_ = np.mean(y)
        self.mu_ = mu
        self.Sigma_ = Sigma
        self.coef_ = _multivariate_normal(mu, Sigma)

    def predict(self, x: npt.NDArray) -> float:
        assert x.shape[1] == self.n_vars, \
            "The number of variables does not match. \
            x has {} variables, but n_vars is {}.".format(x.shape[1], self.n_vars)

        intercept = self.intercept_
        coef = self.coef_

        x = self._order_effects(x)
        return coef @ x.T + intercept

    def _order_effects(self, X: npt.NDArray) -> npt.NDArray:
        return order_effects(X, self.n_vars, self.order)

    def to_qubo(self):
        return make_qubo(self.n_vars, self.coef_)


def _multivariate_normal(mu: npt.NDArray, cov: npt.NDArray) -> npt.NDArray:
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal(cov.shape[0])
    return np.dot(L, z) + mu
