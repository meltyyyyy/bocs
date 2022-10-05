import numpy as np


def make_grid(n_vars: int):
    n_side = np.sqrt(n_vars).astype(int)

    Q = np.zeros((n_vars, n_vars))

    for i in range(n_side):
        for j in range(n_side - 1):
            node = i * n_side + j
            Q[node, node + 1] = 4.95 * np.random.rand() + 0.05
            Q[node + 1, node] = Q[node, node + 1]

    for i in range(n_side - 1):
        for j in range(n_side):
            node = i * n_side + j
            Q[node, node + n_side] = 4.95 * np.random.rand() + 0.05
            Q[node + n_side, node] = Q[node, node + n_side]

    rand_sign = np.tril((np.random.rand(n_vars, n_vars) > 0.5) * 2 - 1, -1)
    rand_sign = rand_sign + rand_sign.T
    return rand_sign * Q
