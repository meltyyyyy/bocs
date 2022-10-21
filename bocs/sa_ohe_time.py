from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot
from aquisitions import simulated_annealing
from sblr import SparseBayesianLinearRegression
import matplotlib.pylab as plt
import numpy.typing as npt
import numpy as np
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = get_logger(__name__)


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = 100, sa_reruns: int = 5, λ: float = 1e+4):
    start = time.time()
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
                surrogate_model, range_vars * n_vars, n_iter=200, sampler=sampler)
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

    X = X[n_init:, :]
    y = y[n_init:]
    end = time.time()
    t = end - start
    return X, y, t


def plot(result: npt.NDArray):
    n_vars = np.arange(5, 16)
    mean = np.mean(result, axis=1)
    var = np.var(result, axis=1)
    std = np.sqrt(np.abs(var))

    fig = plt.figure(figsize=(12, 8))
    plt.yscale('linear')
    plt.xlabel('Number of variables', fontsize=18)
    plt.ylabel('Time', fontsize=18)
    plt.plot(n_vars, mean, label='One Hot Encoding')
    plt.fill_between(n_vars, mean + 2 * std, mean - 2 * std,
                     alpha=.2, label="95% Confidence Interval")
    plt.legend()
    fig.savefig('figs/bocs/sa_ohe_time_10.png')
    plt.close(fig)


def generateExp(n_vars: int):
    s = np.ones(n_vars)
    v = np.ones(n_vars) + 1
    v[-1] = v[-1] + 2
    b = 9
    return {"n_vars": n_vars, "s": s, "v": v, "b": b}


if __name__ == "__main__":
    exps = {}
    for n_vars in range(5, 16, 1):
        exp = generateExp(n_vars)
        exps[f'exp_{n_vars}'] = exp

    # Run Bayesian Optimization
    n_trial = 100
    n_run = 50
    result = np.zeros((len(exps), n_run))

    for i, (key, val) in enumerate(exps.items()):
        n_vars, v, s, b = val['n_vars'], val['v'], val['s'], val['b']

        def objective(X: npt.NDArray, p: float = 2.75) -> npt.NDArray:
            return X @ v.T + p * (b - X @ s.T)

        for j in range(n_run):
            _, _, t = bocs_sa_ohe(objective,
                                  low=0,
                                  high=9,
                                  n_trial=n_trial,
                                  n_vars=n_vars)
            result[i, ] = t
            logger.info(f'n_vars={n_vars}, n_run={j}, time={t}')

    np.save('ohe_time.npy', result)
    plot(result)
