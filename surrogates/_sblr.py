from log import get_sublogger
from utils import order_effects, make_qubo
from typing import Tuple
from scipy.special import comb
import numpy.typing as npt
import numpy as np

logger = get_sublogger(__name__)


class SparseBayesianLinearRegressor:
    def __init__(self, n_vars: int, order: int, random_state: int = 42):
        assert n_vars > 0, "The number of variables must be greater than 0"
        assert order > 0, "order must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.rs = np.random.RandomState(random_state)
        self.n_coef = int(np.sum([comb(n_vars, i) for i in range(order + 1)]))
        self.coef_ = self.rs.normal(0, 1, size=self.n_coef)
        self.intercept_ = 0

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
            "y should be 1 dimension of shape (n_samples, ), but is {}".format(
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
                logger.warn(e)
                continue

            if not np.isnan(_coefs).any():
                needs_sample = 0
        self.intercept_ = _coef0
        self.coef_ = _coefs[:, -1]

    def predict(self, x: npt.NDArray) -> float:
        assert x.shape[1] == self.n_vars, \
            "The number of variables does not match. \
            x has {} variables, but n_vars is {}.".format(x.shape[1], self.n_vars)

        x = self._order_effects(x)
        return self.coef_ @ x.T + self.intercept_

    def _order_effects(self, X: npt.NDArray) -> npt.NDArray:
        return order_effects(X, self.n_vars, self.order)

    def to_qubo(self):
        return make_qubo(self.n_vars, self.coef_)

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
                b = fast_mvgs_(X / sigma, XtX / sigma2, y /
                               sigma, sigma2 * Lambda_star)

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


def fast_mvgs(Phi: npt.NDArray, alpha: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
    """
    Fast sampler for Multivariate Gaussian distributions.

    Applicable for large p, p > n of
    the form N(mu, Σ), where
        mu = Σ Phi^T y
        Σ  = (Phi^T Phi + D^-1)^-1

    Time complexity is O(n^2 p).

    <Reference>
    Fast sampling with Gaussian scale-mixture priors in high-dimensional regression.
    https://arxiv.org/pdf/1506.04778.pdf

    Args:
        Phi (npt.NDArray): Matrix of shape (n, p)
        alpha (npt.NDArray): Array of shape (n, 1)
        D (npt.NDArray): Matrix of shape (p, p)

    Returns:
        npt.NDArray: Array os shape (p, 1)
    """

    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi, u) + delta
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:, np.newaxis])
    w = np.linalg.solve(np.matmul(Phi, Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt, w)

    return x


def fast_mvgs_(Phi: npt.NDArray, PtP: npt.NDArray, alpha: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
    """
    Fast sampler for Multivariate Gaussian distributions.

    Applicable for small p of
    the form N(mu, Σ), where
        mu = Σ Phi' y
        Σ  = (Phi^T Phi + D^-1)^-1

    Time complexity is O(n).

    <Reference>
    Fast sampling of gaussian markov random fields.
    https://arxiv.org/pdf/1506.04778.pdf

    Args:
        Phi (npt.NDArray): Matrix of shape (n, p)
        PtP (npt.NDArray): Matrix of shape (p, p)
        alpha (npt.NDArray): Array of shape (n, 1)
        D (npt.NDArray): Matrix of shape (p, p)

    Returns:
        npt.NDArray: Array os shape (p, 1)
    """

    p = Phi.shape[1]
    D_inv = np.diag(1. / np.diag(D))

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = np.linalg.cholesky(PtP + D_inv)
    except BaseException:
        M = PtP + D_inv
        M = (M + M.T) / 2.
        max_eig = np.max(np.linalg.eigvals(M))
        max_eig = np.real_if_close(max_eig)
        L = np.linalg.cholesky(M + max_eig * 1e-15 * np.eye(M.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T, alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))

    x = m + w

    return x
