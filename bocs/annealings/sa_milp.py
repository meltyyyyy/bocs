import sys
import os
from typing import Tuple
import numpy as np
import numpy.typing as npt
from openjij import SASampler, Response
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from surrogates import BayesianLinearRegressor
from exps import load_study
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits
from pyqubo import Array, Constraint

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "milp"
N_TRIAL = 1000


def bocs_sa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                n_trial: int = N_TRIAL, num_reads: int = 10):
    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegressor(range_vars * n_vars, 2)
    blr.fit(X, y)

    for _ in range(n_trial):
        X_new = []
        qubo = blr.to_qubo()
        while len(X_new) < 5:
            opt_X, _ = simulated_annealing(
                qubo,
                n_vars,
                range_vars,
                num_sweeps=1000,
                num_reads=num_reads)

            for j in range(num_reads):
                if len(X_new) < 5 and np.sum(opt_X[j, :]) == n_vars:
                    X_new.append(opt_X[j, :])

        X_new = np.atleast_2d(X_new)
        y_new = objective(decode_one_hot(low, high, n_vars, X_new))

        # Update posterior
        X = np.vstack((X, X_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        blr.fit(X, y)

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def simulated_annealing(Q: npt.NDArray,
                        n_vars: int,
                        range_vars: int,
                        num_sweeps: int = 1000,
                        num_reads: int = 5,
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

    sampler = SASampler(num_sweeps=num_sweeps, num_reads=num_reads)
    res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sampleset(res)

    opt_X = np.zeros((num_reads, Q.shape[0]))
    opt_y = np.zeros((num_reads,))
    for i in range(num_reads):
        opt_X[i, :] = np.array([samples[i].array('x', j)
                               for j in range(Q.shape[0])])
        opt_y[i] = -1 * samples[i].energy

    return opt_X, -1 * opt_y


def run_bayes_opt(alpha: npt.NDArray,
                  low: int, high: int):

    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    # find global optima
    opt_x = np.atleast_2d(alpha.copy())
    opt_x[opt_x > 0] = high
    opt_x[opt_x < 0] = low
    opt_y = objective(opt_x)

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
    n_vars, low, high, i = int(sys.argv[1]), int(
        sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # load study, extract
    study = load_study(EXP, f'{n_vars}.json')
    alpha = study['alpha']
    n_runs = study['n_runs']

    # run Bayesian Optimization with parallelization
    def runner(i: int): return run_bayes_opt(alpha[i], low, high)
    ans = runner(i)

    filepath = config['output_dir'] + \
        f'annealings/sa/{EXP}/dwave/{n_vars}/{i}_{low}{high}.npy'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, ans)
