import sys
import os
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from aquisitions import simulated_annealing
from surrogates import SparseBayesianLinearRegressor
from exps import find_optimum, load_study
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits
from multiprocessing import Pool

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "miqp"
N_TRIAL = 1000


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = N_TRIAL, sa_reruns: int = 5, λ: float = 10e+8):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegressor(range_vars * n_vars, 2)
    sblr.fit(X, y)

    def penalty(x):
        p = 0
        for i in range(n_vars):
            p += λ * \
                ((1 - np.sum(x[0, i * range_vars: (i + 1) * range_vars])) ** 2)
        return p

    for i in range(n_trial):

        def surrogate_model(x): return sblr.predict(x) - penalty(x)

        sa_X = np.zeros((sa_reruns, range_vars * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(
                surrogate_model,
                range_vars * n_vars,
                cooling_rate=0.99,
                n_iter=100,
                n_flips=1)
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
        sblr.fit(X, y)

        # # log current solution
        x_curr = decode_one_hot(low, high, n_vars, x_new).astype(int)
        y_curr = objective(x_curr)
        logger.info(f'x{i}: {x_curr[0]}, y{i}: {y_curr[0]}')

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def run_bayes_opt(Q: npt.NDArray,
                  low: int, high: int):

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return np.diag(X @ Q @ X.T)

    # find global optima
    n_batch = 2**10
    range_vars = high - low + 1
    if range_vars ** Q.shape[0] % n_batch == 0:
        _, opt_y = find_optimum(objective, low, high,
                                Q.shape[0], n_batch=n_batch)
    else:
        _, opt_y = find_optimum(objective, low, high, Q.shape[0])

    with threadpool_limits(limits=int(os.environ['OPENBLAS_NUM_THREADS']), user_api='blas'):
        _, y = bocs_sa_ohe(objective,
                           low=low,
                           high=high,
                           n_vars=Q.shape[0])
    y = np.maximum.accumulate(y)

    return opt_y - y


if __name__ == "__main__":
    n_vars, low, high = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # load study, extract
    study = load_study(EXP, f'{n_vars}.json')
    Q = np.array(study['Q'])
    n_runs = study['n_runs']
    logger.info(f'experiment: {EXP}, n_vars: {n_vars}')

    # run Bayesian Optimization with parallelization
    def runner(i: int): return run_bayes_opt(Q[i], low, high)
    with Pool(processes=24) as pool:
        imap = pool.imap(runner, range(n_runs))
        data = np.array(list(tqdm(imap, total=n_runs))).T
    filepath = config['output_dir'] + f'{EXP}/time/' + f'ohe_sblr_{n_vars}.npy'
    np.save(filepath, data)
