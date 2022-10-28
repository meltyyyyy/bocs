import numpy as np
import numpy.typing as npt
from typing import Tuple
from utils import sample_binary_matrix, flip_bits
from log import get_logger


def simulated_annealing(objective, n_vars: int, cooling_rate: float = 0.985,
                        n_iter: int = 100, n_flips: int = 1) -> Tuple[npt.NDArray, npt.NDArray]:
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

    X = np.zeros((n_iter, n_vars))
    obj = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    curr_x = sample_binary_matrix(1, n_vars)
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T) + 10e-5

        new_x = flip_bits(curr_x.copy(), n_flips)
        new_obj = objective(new_x)

        # update current solution
        if (new_obj > curr_obj) or (np.random.rand() < np.exp((new_obj - curr_obj) / T)):
            curr_x = new_x
            curr_obj = new_obj

        # Update best solution
        if new_obj > best_obj:
            best_x = new_x
            best_obj = new_obj

        # save solution
        X[i, :] = best_x
        obj[i] = best_obj

    X = X.astype(int)
    return X, obj
