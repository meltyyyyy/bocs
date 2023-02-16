import os
import numpy as np
import numpy.typing as npt
from threadpoolctl import threadpool_limits
from typing import Tuple
from dotenv import load_dotenv
from openjij import SQASampler
from pyqubo import Array, Constraint

load_dotenv()


def simulated_quantum_annealing(Q: npt.NDArray,
                                n_vars: int,
                                range_vars: int,
                                trotter: int = 4,
                                num_sweeps: int = 1000,
                                num_reads: int = 10,
                                λ: float = 10e8) -> Tuple[npt.NDArray, float]:
    """
    Run simulated quantum annealing.

    Simulated Quantum Annealing (SQA) is a Markov Chain Monte-Carlo algorithm
    that samples the equilibrium thermal state of a Quantum Annealing (QA) Hamiltonian.

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

    # sampling
    sampler = SQASampler(trotter=trotter,
                         num_sweeps=num_sweeps,
                         num_reads=num_reads)
    with threadpool_limits(
            limits=int(os.environ["OMP_NUM_THREADS"]),
            user_api='openmp'):
        res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sampleset(res)

    opt_X = np.zeros((len(samples), Q.shape[0]))
    opt_y = np.zeros((len(samples),))
    for i in range(len(samples)):
        opt_X[i, :] = np.array([samples[i].array('x', j)
                               for j in range(Q.shape[0])])
        opt_y[i] = -1 * samples[i].energy

    return opt_X, -1 * opt_y
