import numpy as np
import numpy.typing as npt


def encode_one_hot(low: int, high: int, n_vars: int, X: npt.NDArray) -> npt.NDArray:
    n_samples = X.shape[0]
    range_vars = high - low + 1
    one_hot_X = np.zeros((n_samples, range_vars * n_vars))

    for i in range(n_samples):
        k = 0
        for j in range(n_vars):
            one_hot_X[i][j + n_vars * k] = 1
            k += 1

    return one_hot_X
