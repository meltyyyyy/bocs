import sys
import os
import numpy as np
import numpy.typing as npt
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from aquisitions import simulated_quantum_annealing
from surrogates import BayesianLinearRegressor
from exps import find_optimum, load_study
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "milp"
N_TRIAL = 500


def bocs_sqa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                 n_trial: int = N_TRIAL):
    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegressor(range_vars * n_vars, 2)
    blr.fit(X, y)

    for i in range(n_trial):

        resample = True
        while resample:
            opt_x, _ = simulated_quantum_annealing(
                blr.to_qubo(),
                n_vars,
                range_vars)
            resample = np.sum(opt_x) != n_vars

        # evaluate model objective at new evaluation point
        x_new = np.atleast_2d(opt_x)
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


def run_bayes_opt(alpha: npt.NDArray,
                  low: int, high: int):

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    # find global optima
    opt_x, opt_y = find_optimum(objective, low, high, len(alpha))
    logger.info(f'opt_y: {opt_y}, opt_x: {opt_x}')

    with threadpool_limits(limits=int(os.environ['OPENBLAS_NUM_THREADS']), user_api='blas'):
        _, y = bocs_sqa_ohe(objective,
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
    filepath = config['output_dir'] + f'annealings/sqa/{n_vars}_{low}{high}.npy'
    np.save(filepath, data)
