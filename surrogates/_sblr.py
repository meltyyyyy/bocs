from log import get_logger
from utils import sample_binary_matrix
from sklearn.metrics import mean_squared_error
from itertools import combinations
from typing import Tuple
from scipy.special import comb
from ._fast_mvgs import fast_mvgs, fast_mvgs_
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np

plt.style.use('seaborn-v0_8-pastel')
logger = get_logger(__name__)


class SparseBayesianLinearRegression:
    def __init__(self, n_vars: int, order: int, random_state: int = 42):
        assert n_vars > 0, "The number of variables must be greater than 0"
        assert order > 0, "order must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.rs = np.random.RandomState(random_state)
        self.n_coef = int(np.sum([comb(n_vars, i) for i in range(order + 1)]))
        self.coefs = self.rs.normal(0, 1, size=self.n_coef)

    def fit(self, X: npt.NDArray, y: npt.NDArray):
        """
        Fit Sparse Bayesian Linear Regression.

        Args:
            X (npt.NDArray): matrix of shape (n_samples, n_vars)
            y (npt.NDArray): matrix of shape (n_samples, )
        """
        assert X.shape[1] == self.n_vars,\
            "The number of variables does not match. \
            X has {} variables, but n_vars is {}.".format(X.shape[1], self.n_vars)
        assert y.ndim == 1, \
            "y should be 1 dimension of shape (n_samples, ), but is {}".format(y.ndim)

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)

        needs_sample = 1
        while (needs_sample):
            try:
                _coefs, _coef0 = self._bhs(X, y)
            except Exception as e:
                logger.warn(e)
                continue

            if not np.isnan(_coefs).any():
                needs_sample = 0

        self.coefs = np.append(_coef0, _coefs)

    def predict(self, x: npt.NDArray) -> float:
        assert x.shape[1] == self.n_vars, \
            "The number of variables does not match. \
            x has {} variables, but n_vars is {}.".format(x.shape[1], self.n_vars)

        x = self._order_effects(x)
        return self.coefs[1:] @ x.T + self.coefs[0]

    def _order_effects(self, X: npt.NDArray) -> npt.NDArray:
        """
        Compute order effects.

        Computes data matrix for all coupling
        orders to be added into linear regression model.

        Order is the number of combinations that needs to be taken into consideration,
        usually set to 2.

        Args:
            X (npt.NDArray): input materix of shape (n_samples, n_vars)

        Returns:
            X_allpairs (npt.NDArray): all combinations of variables up to consider,
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

    def _bhs(self, X: npt.NDArray, y: npt.NDArray, n_samples: int = 1,
             burnin: int = 200) -> Tuple[npt.NDArray, float]:
        """
        Run Bayesian Horseshoe Sampler.

        Sample coefficients from conditonal posterior using Gibbs Sampler.

        <Reference>
        A simple sampler for the horseshoe estimator
        https://arxiv.org/pdf/1508.03884.pdf

        Args:
            X (npt.NDArray): input materix of shape (n_samples, 1 + Σ[i=1, order] comb(n_vars, i)).
            y (npt.NDArray): matrix of shape (n_samples, ).
            n_samples (int): The number of sample. Defaults to 1.
            burnin (int): The number of sample to be discarded. Defaults to 200.

        Returns:
            Tuple[np.ndarray, float]: Coefficients for Linear Regression.
        """

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
            sigma = np.sqrt(sigma2)

            if (p > n) and (p > 200):
                b = fast_mvgs(X / sigma, y / sigma, sigma2 * Lambda_star)
            else:
                b = fast_mvgs_(X / sigma, XtX / sigma2, y / sigma, sigma2 * Lambda_star)

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
    sblr = SparseBayesianLinearRegression(n_vars, 2)

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
