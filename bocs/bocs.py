from itertools import product
from typing import Callable
from utils import sample_binary_matrix, get_config
from aquisitions import simulated_annealing, sdp_relaxation
from surrogates import SparseBayesianLinearRegression
import matplotlib.pylab as plt
import numpy.typing as npt
import numpy as np
from exps import load_study
from log import get_logger

config = get_config()
logger = get_logger(__name__, __file__)


def bocs_sa(objective, n_vars: int, n_init: int = 10, n_trial: int = 100, sa_reruns: int = 5, λ: float = 1e-4):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_binary_matrix(n_init, n_vars)
    y = objective(X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    for _ in range(n_trial):

        def surrogate_model(x): return sblr.predict(x) + λ * np.sum(x)
        sa_X = np.zeros((sa_reruns, n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(surrogate_model, n_vars)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = x_new.reshape((1, n_vars))
        y_new = objective(x_new)

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

        # log
        logger.info(f'curr_x: {x_new[0]}, curr_y: {y_new[0]}')

    return X, y


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
        logger.info(f'curr_x: {x_new[0]}, curr_y: {y_new[0]}')

    return X, y


def run_bocs_sa(objective: Callable, n_vars: int, opt_y: int):
    # Run Bayesian Optimization
    X, y = bocs_sa(objective, n_vars)

    n_iter = np.arange(y.size)
    y = np.minimum.accumulate(y)

    # Plot
    fig = plt.figure()
    plt.plot(n_iter, np.abs(y - opt_y))
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Best f(x)')
    filepath = config['output_dir'] + 'bocs_sa.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


def run_bocs_sdp(objective: Callable, n_vars: int, opt_y: int):
    # Run Bayesian Optimization
    X, y = bocs_sdp(objective, n_vars)

    n_iter = np.arange(y.size)
    y = np.minimum.accumulate(y)

    # Plot
    fig = plt.figure()
    plt.plot(n_iter, np.abs(y - opt_y))
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Best f(x)')
    filepath = config['output_dir'] + 'bocs_sdp.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


if __name__ == "__main__":
    n_vars = 10
    experiment = 'bqp'
    study = load_study(experiment, f'{n_vars}.json')
    Q = study['Q']
    logger.info(f'experiment: {experiment}, n_vars: {n_vars}')

    def objective(X: npt.NDArray) -> npt.NDArray:
        return np.diag(X @ Q @ X.T)

    X = np.array(list(map(list, product([0, 1], repeat=n_vars))))
    y = objective(X)

    # find optimal solution
    max_idx = np.argmax(y)
    opt_x = X[max_idx, :]
    opt_y = y[max_idx]
    logger.info(f'opt_x: {opt_x}, opt_y: {opt_y}')

    # fun Bayesian Optimization
    run_bocs_sa(objective, n_vars, opt_y)
    run_bocs_sdp(objective, n_vars, opt_y)
