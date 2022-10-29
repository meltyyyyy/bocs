from datetime import datetime
import os
import sys
from typing import Callable
from dotenv import load_dotenv
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
from exps import load_study
from surrogates import BayesianLinearRegression
from aquisitions import simulated_annealing
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from log import get_logger

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "bqp"


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = 250, sa_reruns: int = 5, λ: float = 10e+8):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegression(range_vars * n_vars, 2)
    blr.fit(X, y)

    def penalty(x):
        p = 0
        for i in range(n_vars):
            p += λ * \
                ((1 - np.sum(x[i * range_vars: (i + 1) + range_vars])) ** 2)
        return p

    for i in range(n_trial):

        def surrogate_model(x): return blr.predict(x) - penalty(x)

        sa_X = np.zeros((sa_reruns, range_vars * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(
                surrogate_model,
                range_vars * n_vars,
                cooling_rate=0.99,
                n_iter=100, n_flips=2)

            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = np.atleast_2d(x_new)
        y_new = objective(decode_one_hot(low, high, n_vars, x_new))

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        blr.fit(X, y)

        # log current solution
        x_curr = decode_one_hot(low, high, n_vars, x_new).astype(int)
        y_curr = objective(x_curr)
        logger.info(f'x{i}: {x_curr[0]}, y{i}: {y_curr[0]}')

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def plot(result: npt.NDArray, opt_y: float, n_vars: int):
    n_iter = np.arange(result.shape[0])
    mean = np.mean(result, axis=1)
    var = np.var(result, axis=1)
    std = np.sqrt(np.abs(var))

    fig = plt.figure(figsize=(12, 8))
    plt.title(f'BQP with {n_vars} variables')
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$f(x_t)$', fontsize=18)
    plt.axhline(opt_y, linestyle="dashed", label='Optimum solution')
    plt.plot(n_iter, mean, label='BOCS + One Hot Encode')
    plt.fill_between(n_iter, mean + 2 * std, mean - 2 * std, alpha=.2)
    plt.legend()
    now = datetime.now()
    filedir = config['output_dir'] + f'{EXP}/' + now.strftime("%m%d") + '/'
    os.makedirs(filedir, exist_ok=True)
    fig.savefig(f'{filedir}' + f'{EXP}_ohe_{n_vars}.png')
    plt.close(fig)


def run_bayes_opt(objective: Callable, low: int, high: int, n_runs: int, n_trial: int = 100):
    result = np.zeros((n_trial, n_runs))

    for i in range(n_runs):
        logger.info(f'############ exp{i} start ############')
        _, y = bocs_sa_ohe(objective,
                           low=low,
                           high=high,
                           n_trial=n_trial,
                           n_vars=n_vars)
        y = np.maximum.accumulate(y)
        result[:, i] = y
        logger.info(f'############  exp{i} end  ############')

    return result


if __name__ == "__main__":
    n_vars, low, high = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    n_vars, low, high = 5, 0, 4

    # load study, extract
    study = load_study(EXP, f'{n_vars}.json')
    Q = study['Q']
    n_runs = study['n_runs']
    optimum = study[f'{low}-{high}']
    opt_x, opt_y = optimum['opt_x'], optimum['opt_y']
    logger.info(f'experiment: {EXP}, n_vars: {n_vars}')
    logger.info(f'opt_x: {opt_x}, opt_y: {opt_y}')

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return np.diag(X @ Q @ X.T)

    # run Bayesian Optimization
    result = run_bayes_opt(objective, low, high, n_runs)

    # save and plot
    plot(result, opt_y, n_vars)
