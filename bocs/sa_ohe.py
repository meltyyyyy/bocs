import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
from sblr import SparseBayesianLinearRegression
from aquisitions import simulated_annealing
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = 100, sa_reruns: int = 5, λ: float = 1e+4):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(range_vars * n_vars, 1)
    sblr.fit(X, y)

    def penalty(x):
        return λ * (n_vars - np.sum(x))

    for _ in range(n_trial):

        def surrogate_model(x): return sblr.predict(x) + penalty(x)
        sa_X = np.zeros((sa_reruns, range_vars * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(surrogate_model, range_vars * n_vars, n_iter=200)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = x_new.reshape((1, range_vars * n_vars))
        y_new = objective(decode_one_hot(low, high, n_vars, x_new))

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

    return X, y


def plot(y: npt.NDArray, true_opt: float):
    n_iter = np.arange(y.size)
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].plot(n_iter, np.maximum.accumulate(y) - true_opt)
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('Optimum - f(x)')
    axes[1].plot(n_iter, np.sort(y) - true_opt)
    axes[1].set_ylabel('Optimum - f(x)')
    fig.tight_layout()
    fig.savefig('figs/bocs_sa_ohe.png')
    plt.close(fig)


def log(X: npt.NDArray, y: npt.NDArray):
    # log
    print("#" * 50)
    print("# Result X and y")
    print("#" * 50)
    print(X)
    print(y)
    print()
    print("#" * 50)
    print("# Sorted y and sorted y - true optimum")
    print("#" * 50)
    print(np.sort(y))
    print(np.sort(y) - true_opt)
    print()
    print("#" * 50)
    print("# Accumulation of y")
    print("#" * 50)
    print(np.maximum.accumulate(y))
    print(np.maximum.accumulate(y) - true_opt)


if __name__ == "__main__":
    n_vars = 5
    s = np.array([1, 1, 1, 1, 1])
    v = np.array([2, 2, 2, 2, 4])
    b = 9
    true_opt = 36

    def objective(X: npt.NDArray, p: float = 2.75) -> npt.NDArray:
        return X @ v.T + p * (b - X @ s.T)

    # Run Bayesian Optimization
    X, y = bocs_sa_ohe(objective, low=0, high=9, n_vars=n_vars)

    plot(y, true_opt)
    log(X, y)
