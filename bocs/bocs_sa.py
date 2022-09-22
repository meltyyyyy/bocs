import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sblr import SparseBayesianLinearRegression
from aquisitions import simulated_annealinng
from utils import sample_binary_matrix


def bocs(objective, n_vars: np.int64, n_init: np.int64 = 10, n_trial: np.int64 = 100, sa_reruns: np.int64 = 5):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_binary_matrix(n_init, n_vars)
    y = objective(X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(n_vars, 2)
    sblr.fit(X, y)

    for i in range(n_trial):

        def surrogate_model(x): return sblr.predict(x)
        sa_X = np.zeros((sa_reruns, n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealinng(surrogate_model, n_vars)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        y_new = objective(x_new)

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

    return X[n_init + 1:, :], y[n_init:]
