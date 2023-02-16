import os
import numpy as np
import numpy.typing as npt
from typing import Tuple
from threadpoolctl import threadpool_limits
from openjij import SASampler
from pyqubo import Array, Constraint
from dotenv import load_dotenv

load_dotenv()


def simulated_annealing(Q: npt.NDArray,
                        n_vars: int,
                        range_vars: int,
                        num_sweeps: int = 1000,
                        num_reads: int = 10,
                        λ: float = 10e8) -> Tuple[npt.NDArray, npt.NDArray]:
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

    # samling
    sampler = SASampler(
        num_sweeps=num_sweeps,
        num_reads=num_reads)
    with threadpool_limits(
            limits=int(os.environ['OPENBLAS_NUM_THREADS']),
            user_api='blas'):
        res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sampleset(res)

    opt_X = np.zeros((len(samples), Q.shape[0]))
    opt_y = np.zeros((len(samples),))
    for i in range(len(samples)):
        opt_X[i, :] = np.array([samples[i].array('x', j)
                               for j in range(Q.shape[0])])
        opt_y[i] = -1 * samples[i].energy

    return opt_X, -1 * opt_y
