from datetime import datetime
import os
import sys
from exps import load_study
import numpy as np
import numpy.typing as npt
from surrogates import BayesianLinearRegressor
from exps import find_optimum
from aquisitions import simulated_annealing
from utils import sample_integer_matrix, encode_binary, decode_binary, get_config, relu
from log import get_logger
from dotenv import load_dotenv

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "milp"
N_TRIAL = 500


def bocs_sa_be(objective, low: int, high: int, n_vars: int, n_init: int = 10,
               n_trial: int = N_TRIAL, sa_reruns: int = 5, λ: float = 10e+8):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Binary expansion
    n_bit = len(bin(high)[2:])
    X = encode_binary(high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegressor(n_bit * n_vars, 2)
    blr.fit(X, y)

    def penalty(x):
        x = decode_binary(high, n_vars, x).astype(int)
        x = relu(x - high) ** 2
        return np.sum(λ * x)

    for i in range(n_trial):

        def surrogate_model(x): return blr.predict(x)

        sa_X = np.zeros((sa_reruns, n_bit * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(
                surrogate_model,
                n_bit * n_vars,
                cooling_rate=0.99,
                n_iter=100,
                n_flips=1)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = np.atleast_2d(x_new)
        y_new = objective(decode_binary(high, n_vars, x_new))

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        blr.fit(X, y)

        # log current solution
        x_curr = decode_binary(high, n_vars, x_new).astype(int)
        y_curr = objective(x_curr)
        logger.info(f'x{i}: {x_curr[0]}, y{i}: {y_curr[0]}')

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def run_bayes_opt(alpha: npt.NDArray,
                  low: int, high: int):

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    # find global optima
    opt_x, opt_y = find_optimum(objective, low, high, len(alpha))
    logger.info(f'opt_y: {opt_y}, opt_x: {opt_x}')

    _, y = bocs_sa_be(objective,
                      low=low,
                      high=high,
                      n_vars=len(alpha))
    y = np.maximum.accumulate(y)

    return opt_y - y


if __name__ == "__main__":
    n_vars, low, high = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # load study, extract
    study = load_study(EXP, f'{n_vars}.json')
    alpha = study['alpha']
    n_runs = study['n_runs']
    logger.info(f'experiment: {EXP}, n_vars: {n_vars}')

    # for store
    data = np.zeros((N_TRIAL, n_runs))

    # run Bayesian Optimization
    for i in range(n_runs):
        logger.info(f'ceofs: {alpha[i]}')
        logger.info(f'############ exp{i} start ############')
        data[:, i] = run_bayes_opt(alpha[i], low, high)
        logger.info(f'############  exp{i} end  ############')

    # save
    now = datetime.now()
    filepath = config['output_dir'] + \
        f'{EXP}/' + now.strftime("%m%d") + '/' + f'be_{n_vars}.npy'
    np.save(filepath, data)
