import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal


def make_qubo(n_vars: int, arr: npt.NDArray) -> npt.NDArray:
    assert n_vars > 1, "The number of variables must be greater than 1."
    assert arr.ndim == 1, "Array must be 1 demension."
    assert arr.size == n_vars + \
        int(n_vars * (n_vars - 1) / 2), "The number of variables does not match."

    Q = np.zeros((n_vars, n_vars))

    # add diagonal
    Q = Q + np.diagflat(arr[:n_vars])

    # create symmetric matrix
    mat = np.zeros((n_vars, n_vars))
    mat[~np.tri(n_vars, dtype=bool, k=0)] = arr[n_vars:]
    mat = (mat + mat.T) / 2
    Q = Q + mat

    assert_array_equal(Q, Q.T)
    return Q
