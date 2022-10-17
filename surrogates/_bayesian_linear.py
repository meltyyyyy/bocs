import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.special import comb
from typing import Tuple
from itertools import combinations
from sklearn.metrics import mean_squared_error
from utils import sample_binary_matrix
from log import get_logger

plt.style.use('seaborn-pastel')
logger = get_logger(__name__)


class BayesianLinearRegression:
    def __init__(self, n_vars: int, order: int, alpha: float = 1e-1, sigma: float = 1e-1, random_state: int = 42):
        assert n_vars > 0, "The number of variables must be greater than 0"
        assert order > 0, "order must be greater than 0"
        assert alpha > 0, "alpha must be greater than 0"
        assert sigma > 0, "sigma must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.alpha_ = alpha
        self.sigma_ = sigma
        self.rs = np.random.RandomState(random_state)
        self.n_coef = int(np.sum([comb(n_vars, i) for i in range(order + 1)]))
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
            "y should be 1 dimension of shape (n_samples, ), but is {}".format(y.ndim)

        sigma = self.sigma_
        alpha = self.alpha_

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)
        XtX = X.T @ X

        # compute covariace
        inner_term = alpha * XtX + sigma * np.eye(X.shape[1])
        Sigma = np.linalg.inv(inner_term)

        # compute mean
        inner_term = XtX + sigma * np.eye(X.shape[1])
        mu = np.linalg.inv(inner_term) @ X.T @ y

        self.intercept_ = np.mean(y)
        self.mu_ = mu
        self.Sigma_ = Sigma

    def predict(self, x: npt.NDArray) -> float:
        assert x.shape[1] == self.n_vars, \
            "The number of variables does not match. \
            x has {} variables, but n_vars is {}.".format(x.shape[1], self.n_vars)

        x = self._order_effects(x)
        x = np.append(1, x)
        return x @ self.coefs

    def _order_effects(self, X: npt.NDArray) -> npt.NDArray:
        """
        Compute order effects.

        Computes data matrix for all coupling
        orders to be added into linear regression model.

        Order is the number of combinations that needs to be taken into consideration,
        usually set to 2.

        Args:
            X (npt.NDArray): Input materix of shape (n_samples, n_vars)

        Returns:
            X_allpairs (npt.NDArray): All combinations of variables up to consider,
                                     which shape is (n_samples, Σ[i=1, order] comb(n_vars, i))
        """
        assert X.shape[1] == self.n_vars,\
            "The number of variables does not match. \
            X has {} variables, but n_vars is {}.".format(X.shape[1], self.n_vars)

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


if __name__ == '__main__':
    n_vars = 10
    rs = np.random.RandomState(42)
    Q: npt.NDArray = rs.randn(n_vars**2).reshape(n_vars, n_vars)

    def objective(X: npt.NDArray) -> npt.NDArray:
        return np.diag(X @ Q @ X.T)

    X_train = sample_binary_matrix(10, n_vars)
    y_train = objective(X_train)
    X_test = sample_binary_matrix(100, n_vars)
    y_test = objective(X_test)

    # with 2 order
    sblr = BayesianLinearRegression(n_vars, 2)

    loss = []
    for _ in range(90):
        x_new = sample_binary_matrix(1, n_vars)
        y_new = objective(x_new)
        X_train = np.vstack((X_train, x_new))
        y_train = np.hstack((y_train, y_new))

        # train, predict, evaluate
        sblr.fit(X_train, y_train)
        y_pred = np.array([sblr.predict(x.reshape(1, n_vars)) for x in X_test])
        mse = mean_squared_error(y_test, y_pred)
        loss.append(mse)

    # plot
    fig = plt.figure()
    plt.plot(loss)
    plt.xlabel('number of samples')
    plt.ylabel('loss')
    fig.savefig('sblr.png')
    plt.close()
