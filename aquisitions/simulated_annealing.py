import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from surrogates import SparseBayesianLinearRegression
from utils import sample_binary_matrix


def simulated_annealing(objective, n_vars: int, cooling_rate: float = 0.985,
                        n_iter: int = 100, sampler: Callable[[int], npt.NDArray] = None) -> Tuple[npt.NDArray, npt.NDArray]:
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

    if sampler is None:
        def sampler(n): return sample_binary_matrix(n, n_vars)

    X = np.zeros((n_iter, n_vars))
    obj = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    curr_x = sampler(1)
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T)

        new_x = sampler(1)
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

    return X, obj


if __name__ == "__main__":
    rs = np.random.RandomState(42)
    n_vars = 10
    Q = rs.randn(n_vars**2).reshape(n_vars, n_vars)

    def objective(X: np.ndarray) -> np.ndarray:
        return np.diag(X @ Q @ X.T)

    X = sample_binary_matrix(10, n_vars)
    y = objective(X)

    # with 2 order
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    def stat_model(x): return sblr.predict(x)
    sa_X, sa_obj = simulated_annealing(stat_model, n_vars)

    # plot
    fig = plt.figure()
    plt.plot(sa_obj)
    plt.xlabel('number of iteration')
    plt.ylabel('objective')
    fig.savefig('sa.png')
    plt.close()
