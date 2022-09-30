import numpy as np
import numpy.typing as npt

# TODO : remove low since input should be greater than 0


def encode_one_hot(low: int, high: int, n_vars: int, X: npt.NDArray) -> npt.NDArray:
    n_samples = X.shape[0]
    range_vars = high - low + 1
    assert X.ndim == 2, "X needs to be at least 2d."
    assert X.shape[1] % n_vars == 0, "Inconsistent dimensions."
    assert n_vars == X.shape[1], "The number of variable does not match."

    one_hot_X = np.zeros((n_samples, range_vars * n_vars))

    for i in range(n_samples):
        k = 0
        for j in range(n_vars):
            one_hot_X[i, int(X[i, j] + range_vars * k)] = 1
            k += 1

    return one_hot_X


def encode_binary(high: int, n_vars: int, X: npt.NDArray) -> npt.NDArray:
    n_samples = X.shape[0]
    assert X.ndim == 2, "X needs to be at least 2d."
    assert X.shape[1] % n_vars == 0, "Inconsistent dimensions."
    assert n_vars == X.shape[1], "The number of variable does not match."
    assert np.all(X >= 0), "X should be all integer greater than 0."

    b = bin(high)[2:]
    n_bit = len(b)
    binary_X = np.zeros((n_samples, n_bit * n_vars))

    # float -> int
    X = X.astype(int)

    for i in range(n_samples):
        x = []
        for j in range(n_vars):
            b = bin(X[i, j])[2:]
            diff_bit = n_bit - len(b)
            assert diff_bit >= 0, "Each element of X should be smaller than high."
            while diff_bit > 0:
                b = '0' + b
                diff_bit -= 1
            x = x + list(reversed(list(map(int, list(b)))))

        binary_X[i, :] = np.array(x)

    return binary_X
