import numpy as np


def knapsack(n_vars: int):
    v = np.abs(np.random.randn(n_vars))
    w = np.abs(np.random.randn(n_vars))
    w_max = n_vars * np.random.randn()
    return v, w, w_max
