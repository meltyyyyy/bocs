from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from itertools import combinations
from dwave.system import EmbeddingComposite, DWaveSampler
from log import get_sublogger

logger = get_sublogger(__name__)


def quantum_annealing(Q: npt.NDArray,
                      sampler: Union[EmbeddingComposite, DWaveSampler],
                      n_vars: int,
                      range_vars: int,
                      num_reads: int = 10,
                      λ: float = 10e8) -> Tuple[npt.NDArray, float]:
    """
    Run quantum annealing.


    Args:
        coef :

    Returns:
        Tuple[npt.NDArray, npt.NDArray]:
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

    # sampling
    res = sampler.sample_qubo(Q=qubo, num_reads=num_reads)
    samples = sorted(res.record, key=lambda x: x[1])

    opt_X = np.zeros((num_reads, Q.shape[0]))
    opt_y = np.zeros((num_reads, ))
    for i in range(num_reads):
        opt_X[i, :] = samples[i][0]
        opt_y[i] = -1 * samples[i][1]

    return opt_X, -1 * opt_y
