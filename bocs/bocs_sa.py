import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pylab as plt
from sblr import SparseBayesianLinearRegression
from aquisitions import simulated_annealinng
from utils import sample_binary_matrix

rs = np.random.RandomState(42)


def bocs_sa(objective, n_vars: np.int64, n_init: np.int64 = 10, n_trial: np.int64 = 100, sa_reruns: np.int64 = 5):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_binary_matrix(n_init, n_vars)
    y = objective(X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    for _ in range(n_trial):

        def surrogate_model(x): return sblr.predict(x)
        sa_X = np.zeros((sa_reruns, n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealinng(surrogate_model, n_vars)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = x_new.reshape((1, 15))
        y_new = objective(x_new)

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

    return X, y


def quad_matrix(n_vars, alpha):
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

    def objective(X: np.ndarray) -> np.float64:
        return - np.diag(X @ Q @ X.T)

    # Run Bayesian Optimization
    X, y = bocs_sa(objective, n_vars)

    n_iter = np.arange(y.size)
    bocs_opt = np.minimum.accumulate(y)
    y_opt = np.min(objective(sample_binary_matrix(1000, n_vars)))

    # Plot
    fig = plt.figure()
    plt.plot(n_iter, np.abs(bocs_opt - y_opt))
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Best f(x)')
    fig.savefig('bocs_sa.png')
    plt.close(fig)
