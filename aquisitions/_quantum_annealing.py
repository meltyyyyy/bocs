import os
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from pyqubo import Array, Constraint
from dwave.system import EmbeddingComposite, DWaveSampler
from log import get_sublogger
from threadpoolctl import threadpool_limits

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

    # sampling
    res = sampler.sample_qubo(Q=qubo, num_reads=num_reads)
    samples = model.decode_sampleset(res)

    opt_X = np.zeros((len(samples), Q.shape[0]))
    opt_y = np.zeros((len(samples),))
    for i in range(len(samples)):
        opt_X[i, :] = np.array([samples[i].array('x', j)
                               for j in range(Q.shape[0])])
        opt_y[i] = -1 * samples[i].energy

    return opt_X, -1 * opt_y
