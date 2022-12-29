import sys
import os
from typing import Tuple
import numpy as np
import numpy.typing as npt
from openjij import SASampler
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from surrogates import BayesianLinearRegressor
from exps import find_optimum, load_study
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits
from pyqubo import Array, Constraint
from multiprocessing import Pool
from tqdm import tqdm


load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "milp"
N_TRIAL = 1500


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = N_TRIAL, sa_reruns: int = 4, λ: float = 10e+8):
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
        sa_X = np.zeros((sa_reruns, range_vars * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            resample = True
            qubo = blr.to_qubo()
            while resample:
                opt_x, opt_y = simulated_annealing(
                    qubo,
                    n_vars,
                    range_vars,
                    num_sweeps=1000)
                resample = np.sum(opt_x) != n_vars

            sa_X[j, :] = opt_x
            sa_y[j] = opt_y

        # evaluate model objective at new evaluation point
        max_idx = np.argmax(sa_y)
        x_new = np.atleast_2d(sa_X[max_idx, :])
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


def simulated_annealing(Q: npt.NDArray,
                        n_vars: int,
                        range_vars: int,
                        num_sweeps: int = 1000,
                        λ: float = 10e8) -> Tuple[npt.NDArray, float]:
    """
    Run simulated annealing.

    Simulated Annealing (SA) is a probabilistic technique
    for approximating the global optimum of a given function.

    Args:
        ---: ---

    Returns:
        ---: ---
    """
    assert Q.ndim == 2
    assert Q.shape[0] == Q.shape[1]

    # define objective
    x = Array.create('x', shape=(n_vars * range_vars, ), vartype='BINARY')
    Q = Array(Q)
    H_A = Constraint(sum(λ * (1 - sum(x[j * range_vars + i] for i in range(range_vars)))
                     ** 2 for j in range(n_vars)), label='HA')
    H_B = x @ Q @ x.T
    H = H_A - H_B

    # define QUBO
    model = H.compile()
    qubo, _ = model.to_qubo()

    sampler = SASampler(num_sweeps=num_sweeps)

    res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sample(res.first.sample, vartype="BINARY")

    opt_x = np.array([samples.array('x', i) for i in range(Q.shape[0])])
    opt_y = samples.energy

    return opt_x, -1 * opt_y


def run_bayes_opt(alpha: npt.NDArray,
                  low: int, high: int):

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    # find global optima
    # opt_x, opt_y = find_optimum(objective, low, high, len(alpha))
    opt_x = np.atleast_2d(alpha.copy())
    opt_x[opt_x > 0] = high
    opt_x[opt_x < 0] = low
    opt_y = objective(opt_x)
    logger.info(f'opt_y: {opt_y}, opt_x: {opt_x}')

    with threadpool_limits(
            limits=int(os.environ['OPENBLAS_NUM_THREADS']),
            user_api='blas'):

        _, y = bocs_sa_ohe(objective,
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

    # run Bayesian Optimization with parallelization
    def runner(i: int): return run_bayes_opt(alpha[i], low, high)
    with Pool(processes=50) as pool:
        imap = pool.imap(runner, range(n_runs))
        data = np.array(list(tqdm(imap, total=n_runs))).T

    filepath = config['output_dir'] + f'annealings/sa/{n_vars}_{low}{high}.npy'
    np.save(filepath, data)
