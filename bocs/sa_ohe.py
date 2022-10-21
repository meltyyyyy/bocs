from typing import Callable
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
from exps import load_study
from surrogates import SparseBayesianLinearRegression
from aquisitions import simulated_annealing
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot
from log import get_logger
from itertools import product

logger = get_logger(__name__, __file__)


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

    for i in range(n_trial):

        def surrogate_model(x): return sblr.predict(x) + penalty(x)

        # Sampler for new x in Simulated Annealing.
        # Probability of success corresponds to constraint for one hot encoding.
        # ex.
        # range_vars = 10 -> p = 0.1
        def sampler(n: int) -> npt.NDArray:
            samples = np.random.binomial(
                n, p=1 / range_vars, size=range_vars * n_vars)
            return np.atleast_2d(samples)

        sa_X = np.zeros((sa_reruns, range_vars * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(
                surrogate_model, range_vars * n_vars, n_iter=200)
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

        # log current solution
        x_curr = decode_one_hot(low, high, n_vars, x_new).astype(int)
        y_curr = objective(x_curr)
        logger.info(f'x{i}: {x_curr[0]}, y{i}: {y_curr[0]}')

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def plot(result: npt.NDArray, opt_y: float):
    n_iter = np.arange(result.shape[0])
    mean = np.mean(result, axis=1)
    var = np.var(result, axis=1)
    std = np.sqrt(np.abs(var))

    fig = plt.figure(figsize=(12, 8))
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$f(x_t)$', fontsize=18)
    plt.axhline(opt_y, linestyle="dashed", label='Optimum solution')
    plt.plot(n_iter, mean, label='BOCS + One Hot Encode')
    plt.fill_between(n_iter, mean + 2 * std, mean - 2 * std, alpha=.2)
    plt.legend()
    fig.savefig('figs/bocs/sa_ohe_10.png')
    plt.close(fig)


def find_optimum(objective: Callable, low: int, high: int, n_vars: int, n_batch: int = 1):
    range_vars = high - low + 1
    assert range_vars ** n_vars < 2 ** 32, "The number of combinations for variables is too large."
    assert range_vars ** n_vars % n_batch == 0, "The number of combinations for variables must be divided by batch_size."

    # Generate all cases
    X = np.array(list(map(list, product(
        np.arange(low, high + 1).tolist(), repeat=n_vars))), dtype=np.int8)
    y = np.zeros(range_vars ** n_vars, dtype=np.float16)

    # Split X into batches
    batches = np.split(X, n_batch, axis=0)
    batch_size = range_vars ** n_vars // n_batch
    for i, X_batch in enumerate(batches):
        y[i * batch_size: (i + 1) * batch_size] = objective(X_batch)

    # Find optimal solution
    max_idx = np.argmax(y)
    opt_x = X[max_idx, :]
    opt_y = y[max_idx]
    del y, batches

    return opt_x, opt_y


def run_bayes_opt(n_runs: int, n_trial: int = 100):
    result = np.zeros((n_trial, n_runs))

    for i in range(n_runs):
        logger.info(f'exp{i} start')
        _, y = bocs_sa_ohe(objective,
                           low=0,
                           high=4,
                           n_trial=n_trial,
                           n_vars=n_vars)
        y = np.maximum.accumulate(y)
        result[:, i] = y
        logger.info(f'exp{i} end')


if __name__ == "__main__":
    n_vars = 10
    experiment = 'bqp'
    study = load_study(experiment, f'{n_vars}.json')
    Q = study['Q']
    Q = Q.astype(np.float16)
    n_runs = study['n_runs']
    logger.info(f'experiment: {experiment}, n_vars: {n_vars}')
    n_runs = 2

    def objective(X: npt.NDArray) -> npt.NDArray:
        return X @ Q @ X.T

    opt_x, opt_y = find_optimum(objective, 0, 4, n_vars, n_batch=5 ** 3)

    # Run Bayesian Optimization
    n_trial = 100
    result = np.zeros((n_trial, n_runs))

    # np.save(STUDY_DIR + experiment + '/' + f'{n_vars}.npy', result)
    # plot(result, opt_y)
