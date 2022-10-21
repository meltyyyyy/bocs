from exps import sbqp
from utils import sample_binary_matrix
from aquisitions import sdp_relaxation
from sblr import SparseBayesianLinearRegression
import matplotlib.pylab as plt
import numpy.typing as npt
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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


if __name__ == "__main__":
    n_vars = 10
    Q = sbqp(n_vars, 10)

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
