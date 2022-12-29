import numpy as np
import numpy.typing as npt
from typing import Tuple
from dotenv import load_dotenv
from openjij import SQASampler, SASampler
from pyqubo import Array, Constraint

load_dotenv()


def simulated_quantum_annealing(Q: npt.NDArray,
                                n_vars: int,
                                range_vars: int,
                                trotter: int = 1,
                                num_sweeps: int = 1,
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

    sampler = SQASampler(trotter=trotter, num_sweeps=num_sweeps)

    res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sample(res.first.sample, vartype="BINARY")

    opt_x = np.array([samples.array('x', i) for i in range(Q.shape[0])])
    opt_y = samples.energy

    return opt_x, -1 * opt_y
