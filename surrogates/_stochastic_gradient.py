import numpy as np
import numpy.typing as npt
from scipy.special import comb
from utils import order_effects, make_qubo


class SGDRegressor:
    def __init__(self, n_vars: int, order: int, alpha: float = 1e-2, max_iter: int = 10e4, random_state: int = 42):
        assert n_vars > 0, "The number of variables must be greater than 0."
        assert order > 0, "order must be greater than 0."
        assert alpha > 0, "alpha must be greater than 0."
        assert max_iter > 0, 'iteration must be greater than 0.'
        self.n_vars = n_vars
        self.order = order
        self.alpha_ = alpha
        self.max_iter_ = int(max_iter)
        self.rs = np.random.RandomState(random_state)
        self.n_coef_ = int(np.sum([comb(n_vars, i)
                           for i in range(1, order + 1)]))
        self.intercept_ = 0
        self.coef_ = self.rs.randn(self.n_coef_)

    def fit(self, X: npt.NDArray, y: npt.NDArray, batch_size: int, early_stopping_rounds: int = 100):
        """
        Fit SGDRegressor.
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
        assert X.shape[0] > batch_size, f"batch_size should be greater than {X.shape[0]}"

        self.intercept_ = np.mean(y)
        coef_ = self.coef_
        n_coef_ = self.n_coef_
        alpha_ = self.alpha_
        n_samples_ = X.shape[0]

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)
        assert self.n_coef_, X.shape[1]

        X = X
        min_loss = np.inf
        rounds = 0

        for _ in range(0, self.max_iter_):
            pred = coef_ @ X.T
            loss = np.sum((pred - y) ** 2) / 2

            # for early stopping rounds
            if min_loss < loss:
                rounds += 1
                if early_stopping_rounds == rounds:
                    break
            min_loss = min(min_loss, loss)

            # random index
            indices = np.random.randint(0, n_samples_, size=batch_size)
            grad_ = np.zeros(n_coef_, dtype=np.float64)
            for index in indices.tolist():
                grad_ += (coef_ @ X[index, :].T - y[index]) * X[index, :]

            coef_ = coef_ - alpha_ * grad_

        self.coef_ = coef_

    def predict(self, x: npt.NDArray) -> float:
        assert x.shape[1] == self.n_vars, \
            "The number of variables does not match. \
            x has {} variablZes, but n_vars is {}.".format(x.shape[1], self.n_vars)

        coef_ = self.coef_
        intercept = self.intercept_

        x = self._order_effects(x)
        return coef_ @ x.T + intercept

    def _order_effects(self, X: npt.NDArray) -> npt.NDArray:
        return order_effects(X, self.n_vars, self.order)

    def to_qubo(self):
        return make_qubo(self.n_vars, self.coef_)
