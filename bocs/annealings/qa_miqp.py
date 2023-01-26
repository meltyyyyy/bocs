import sys
import os
from typing import Tuple
import numpy as np
import numpy.typing as npt
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot, get_config
from surrogates import BayesianLinearRegressor
from exps import load_study, find_optimum
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits
from pyqubo import Array, Constraint
from dwave.system import DWaveSampler, EmbeddingComposite

load_dotenv()
config = get_config()
logger = get_logger(__name__, __file__)
EXP = "miqp"
N_TRIAL = 1000


def bocs_sqa_ohe(objective, low: int, high: int, n_vars: int, n_init: int = 10,
                 n_trial: int = N_TRIAL, num_add: int = 5):
    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegressor(range_vars * n_vars, 2)
    blr.fit(X, y)

    # define sampler
    sampler = EmbeddingComposite(DWaveSampler(
        solver='DW_2000Q_6',
        token=os.environ['DWAVE_TOKEN'],
        endpoint=os.environ["DWAVE_API_ENDPOINT"]))

    for i in range(n_trial):
        X_new = []
        qubo = blr.to_qubo()
        while len(X_new) < num_add:
            opt_X, _ = simulated_quantum_annealing(
                qubo,
                sampler,
                n_vars,
                range_vars)

            for j in range(num_add):
                if len(X_new) < num_add and np.sum(opt_X[j, :]) == n_vars:
                    X_new.append(opt_X[j, :])

        # evaluate model objective at new evaluation point
        X_new = np.atleast_2d(X_new)
        y_new = objective(decode_one_hot(low, high, n_vars, X_new))

        # Update posterior
        X = np.vstack((X, X_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        blr.fit(X, y)

        logger.info(f"iteration {i} done")

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def simulated_quantum_annealing(Q: npt.NDArray,
                                sampler: EmbeddingComposite,
                                n_vars: int,
                                range_vars: int,
                                num_reads: int = 10,
                                λ: float = 10e8) -> Tuple[npt.NDArray, float]:
    """
    Run simulated quantum annealing.

    Simulated Quantum Annealing (SQA) is a Markov Chain Monte-Carlo algorithm
    that samples the equilibrium thermal state of a Quantum Annealing (QA) Hamiltonian.
    num_sweepsの1ステップ定義

    Args:
        coef : objective function / statistical model

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: Best solutions that maximize objective.
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

    res = sampler.sample_qubo(Q=qubo, num_reads=num_reads)
    samples = model.decode_sampleset(res)

    opt_X = np.zeros((len(samples), Q.shape[0]))
    opt_y = np.zeros((len(samples),))
    for i in range(len(samples)):
        opt_X[i, :] = np.array([samples[i].array('x', j)
                               for j in range(Q.shape[0])])
        opt_y[i] = -1 * samples[i].energy

    return opt_X, -1 * opt_y


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

    with threadpool_limits(
            limits=int(os.environ['OPENBLAS_NUM_THREADS']),
            user_api='blas'):
        _, y = bocs_sqa_ohe(objective,
                            low=low,
                            high=high,
                            n_vars=Q.shape[0])
    y = np.maximum.accumulate(y)

    return opt_y - y


if __name__ == "__main__":
    n_vars, low, high, i = int(sys.argv[1]), int(
        sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # load study, extract
    study = load_study(EXP, f'{n_vars}.json')
    Q = np.array(study['Q'])
    n_runs = study['n_runs']
    logger.info(f'experiment: {EXP}, n_vars: {n_vars}')

    # run Bayesian Optimization with parallelization
    def runner(i: int): return run_bayes_opt(Q[i], low, high)
    ans = runner(i)

    filepath = config['output_dir'] + \
        f'annealings/sqa/{EXP}/qpu/{n_vars}/{i}_{low}{high}.npy'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, ans)
