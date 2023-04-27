import os
import numpy as np
import numpy.typing as npt
from typing import Tuple
from threadpoolctl import threadpool_limits
from openjij import SASampler
from dotenv import load_dotenv
from itertools import combinations

load_dotenv()


def simulated_annealing(Q: npt.NDArray,
                        n_vars: int,
                        range_vars: int,
                        num_sweeps: int = 1000,
                        num_reads: int = 10,
                        λ: float = 10e4) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Run simulated annealing.

    Simulated Annealing (SA) is a probabilistic technique
    for approximating the global optimum of a given function.

    Args:
        objective : objective function / statistical model
        n_vars (int): The number of variables
        cooling_rate (float): Defaults to 0.985.
        n_iter (int): The number of iterations for SA. Defaults to 100.
        sampler (Callable[[int], npt.NDArray], optional): Sampler for new x.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: Best solutions that maximize objective.
    """
    assert Q.ndim == 2
    assert Q.shape[0] == Q.shape[1]

    # ----- Constraint -----
    constraint = {}
    for i in range(n_vars * range_vars):
        constraint[(i, i)] = -1 * λ
    for i in range(n_vars):
        for c in list(combinations(list(range(i * range_vars, (i + 1) * range_vars)), 2)):
            constraint[(c[0], c[1])] = 2 * λ

    # ----- QUBO -----
    QUBO = {}
    for i in range(n_vars * range_vars):
        QUBO[(i, i)] = Q[i, i]
    for i in range(n_vars * range_vars):
        for j in range(i + 1, n_vars * range_vars):
            QUBO[(i, j)] = 2 * Q[i, j]

    qubo = {}
    for i, j in QUBO.keys():
        qubo[(i, j)] = -1 * QUBO[(i, j)]
        if (i, j) in constraint.keys():
            qubo[(i, j)] += constraint[(i, j)]

    # samling
    sampler = SASampler(
        num_sweeps=num_sweeps,
        num_reads=num_reads)
    with threadpool_limits(
            limits=int(os.environ['OPENBLAS_NUM_THREADS']),
            user_api='blas'):
        res = sampler.sample_qubo(Q=qubo)
    samples = sorted(res.record, key=lambda x: x[1])

    opt_X = np.zeros((num_reads, Q.shape[0]))
    opt_y = np.zeros((num_reads, ))
    for i in range(num_reads):
        opt_X[i, :] = samples[i][0]
        opt_y[i] = -1 * samples[i][1]

    return opt_X, -1 * opt_y
