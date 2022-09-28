import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sblr import SparseBayesianLinearRegression
from utils import sample_binary_matrix
rs = np.random.RandomState(42)


def simulated_annealing(objective, n_vars: int, cooling_rate: float = 0.985,
                        n_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Run simulated annealing
    Simulated Annealing (SA) is a probabilistic technique
    for approximating the global optimum of a given function.

    Args:
        objective : objective function / statistical model
        n_vars (np.int64): The number of variables
        cooling_rate (np.float64, optional): Defaults to 0.985.
        n_iter (np.int64, optional): Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Best solutions that maximize objective.
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
        T = cool(T)

        new_x = sample_binary_matrix(1, n_vars)
        new_obj = objective(new_x)

        # update current solution
        if (new_obj > curr_obj) or (rs.rand() < np.exp((new_obj - curr_obj) / T)):
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
