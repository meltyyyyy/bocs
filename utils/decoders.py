import numpy as np
import numpy.typing as npt


def decode_one_hot(low: int, high: int, n_vars: int, X: npt.NDArray):
    n_samples = X.shape[0]
    range_vars = high - low + 1

    assert X.ndim == 2, "X needs to be at least 2d."
    assert X.shape[1] % n_vars == 0, "Inconsistent dimensions."
    assert range_vars * n_vars == X.shape[1], "The number of variable does not match."

    decimal_X = np.zeros((n_samples, n_vars))
    decimals = np.arange(low, high + 1)

    for i in range(n_samples):
        k = 0
        x = np.zeros(n_vars)
        for j in range(n_vars):
            x[j] = X[i, k * range_vars: (k + 1) * range_vars] @ decimals.T
            k += 1

        decimal_X[i, :] = x

    return decimal_X
