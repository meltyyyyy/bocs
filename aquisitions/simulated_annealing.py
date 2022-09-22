from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from sblr.sblr import SparseBayesianLinearRegression
from utils.sampling import sample_binary_matrix

rs = np.random.RandomState(42)


def simulated_annealinng(objective, n_vars: np.int64, cooling_rate: np.float64 = 0.985,
                         n_iter: np.int64 = 100) -> Union[np.ndarray, np.ndarray]:
    """Run simulated annealing
    Simulated Annealing (SA) is a probabilistic technique
    for approximating the global optimum of a given function.

    Args:
        objective : objective function / statistical model
        n_vars (np.int64): The number of variables
        cooling_rate (np.float64, optional): Defaults to 0.985.
        n_iter (np.int64, optional): Defaults to 100.

    Returns:
        Union[np.ndarray, np.ndarray]: Best solutions that maximize objective.
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
        if (new_obj > curr_obj) or (rs.rand()
                                    < np.exp((new_obj - curr_obj) / T)):
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

    def objective(X: np.ndarray) -> np.float64:
        return np.diag(X @ Q @ X.T)

    X = sample_binary_matrix(10, n_vars)
    y = objective(X)

    # with 2 order
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    stat_model = lambda x: sblr.predict(x)
    sa_X, sa_obj = simulated_annealinng(stat_model, n_vars)

    # plot
    fig = plt.figure()
    plt.plot(sa_obj)
    plt.xlabel('number of iteration')
    plt.ylabel('objective')
    fig.savefig('sa.png')
    plt.close()
