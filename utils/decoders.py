import numpy as np
import numpy.typing as npt


def decode_one_hot(low: int, high: int, n_vars: int, X: npt.NDArray):
    n_samples = X.shape[0]
    range_vars = high - low + 1

    assert X.ndim == 2, "X needs to be at least 2d."
    assert range_vars * n_vars == X.shape[1], "The number of variable does not match."
    assert np.all((X == 0) | (X == 1)), "X should be binary matrix."

    decimal_X = np.zeros((n_samples, n_vars))
    radix = np.arange(low, high + 1)

    for i in range(n_samples):
        k = 0
        x = np.zeros(n_vars)
        for j in range(n_vars):
            x[j] = X[i, k * range_vars: (k + 1) * range_vars] @ radix.T
            k += 1

        decimal_X[i, :] = x

    return decimal_X


def decode_binary(high: int, n_vars: int, X: npt.NDArray):
    n_samples = X.shape[0]

    assert X.ndim == 2, "X needs to be at least 2d."
    assert X.shape[1] % n_vars == 0, "The number of variable does not match."
    assert np.all((X == 0) | (X == 1)), "X should be binary matrix."

    b = bin(high)[2:]
    n_bit = len(b)
    radix = np.array([2 ** i for i in range(n_bit)])

    decimal_X = np.zeros((n_samples, n_vars))

    for i in range(n_samples):
        k = 0
        x = np.zeros(n_vars)
        for j in range(n_vars):
            x[j] = X[i, k * n_bit: (k + 1) * n_bit] @ radix.T
            k += 1

        decimal_X[i, :] = x

    return decimal_X
