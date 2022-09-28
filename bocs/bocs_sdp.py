import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
from sblr import SparseBayesianLinearRegression
from aquisitions import sdp_relaxation
from utils import sample_binary_matrix

rs = np.random.RandomState(42)


def bocs_sdp(objective, n_vars: int, n_init: int = 10, n_trial: int = 100):
    # Initial samples
    X = sample_binary_matrix(n_init, n_vars)
    y = objective(X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    for _ in range(n_trial):
        x_new, _ = sdp_relaxation(sblr.coefs, n_vars)

        # evaluate model objective at new evaluation point
        x_new = x_new.reshape((1, n_vars))
        y_new = objective(x_new)

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

    return X, y


def quad_matrix(n_vars: int, alpha: int) -> npt.NDArray:
    i = np.linspace(1, n_vars, n_vars)
    j = np.linspace(1, n_vars, n_vars)

    def K(s, t): return np.exp(-1 * (s - t)**2 / alpha)
    decay = K(i[:, None], j[None, :])

    Q = np.random.randn(n_vars, n_vars)
    Q = Q * decay

    return Q


if __name__ == "__main__":
    n_vars = 10
    Q = quad_matrix(n_vars, 10)

    def objective(X: npt.NDArray) -> npt.NDArray:
        return - np.diag(X @ Q @ X.T)

    # Run Bayesian Optimization
    X, y = bocs_sdp(objective, n_vars)

    n_iter = np.arange(y.size)
    bocs_opt = np.minimum.accumulate(y)
    y_opt = np.min(objective(sample_binary_matrix(1000, n_vars)))

    # Plot
    fig = plt.figure()
    plt.plot(n_iter, np.abs(bocs_opt - y_opt))
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Best f(x)')
    fig.savefig('bocs_sdp.png')
    plt.close(fig)
