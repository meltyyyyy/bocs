import numpy as np
import numpy.typing as npt
from typing import Tuple
from dotenv import load_dotenv
from openjij import SQASampler
from pyqubo import Array, Constraint

load_dotenv()


def simulated_quantum_annealing(coef: npt.NDArray, n_vars: int, range_vars: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Run simulated quantum annealing.

    Simulated Quantum Annealing (SQA) is a Markov Chain Monte-Carlo algorithm
    that samples the equilibrium thermal state of a Quantum Annealing (QA) Hamiltonian.

    Args:
        coef : objective function / statistical model
        cooling_rate (float): Defaults to 0.985.
        n_iter (int): The number of iterations for SA. Defaults to 100.
        sampler (Callable[[int], npt.NDArray], optional): Sampler for new x.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: Best solutions that maximize objective.
    """
    assert coef.ndim == 1

    # define objective
    x = Array.create('x', shape=(coef.shape[0],), vartype='BINARY')
    H_A = Constraint(sum((1 - sum(x[j * range_vars + i] for i in range(range_vars)))**2 for j in range(n_vars)),label='HA')
    H_B = sum(coef[i] * x[i] for i in range(coef.shape[0]))
    Q = H_A + H_B

    # define QUBO
    model = Q.compile()
    qubo, offset = model.to_qubo()

    sampler = SQASampler()
    res = sampler.sample_qubo(Q=qubo)
    samples = model.decode_sample(res.first.sample, vartype="BINARY")
    x = np.array([])
    for i in range(coef.shape[0]):
        x.append(samples.array('x', i))

    return x
