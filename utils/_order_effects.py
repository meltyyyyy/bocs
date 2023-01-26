import numpy as np
import numpy.typing as npt
from itertools import combinations


def order_effects(X: npt.NDArray, n_vars: int, order: int) -> npt.NDArray:
    """
    Compute order effects.

    Computes data matrix for all coupling
    orders to be added into linear regression model.

    Order is the number of combinations that needs to be taken into consideration,
    usually set to 2.

    Args:
        X (npt.NDArray): input materix of shape (n_samples, n_vars)

    Returns:
        X_allpairs (npt.NDArray): all combinations of variables up to consider,
                                 which shape is (n_samples, Σ[i=1, order] comb(n_vars, i))
    """
    assert X.shape[1] == n_vars,\
        "The number of variables does not match. \
            X has {} variables, but n_vars is {}.".format(X.shape[1], n_vars)

    n_samples, n_vars = X.shape
    X_allpairs = X.copy()

    for i in range(2, order + 1, 1):

        # generate all combinations of indices (without diagonals)
        offdProd = np.array(list(combinations(np.arange(n_vars), i)))

        # generate products of input variables
        x_comb = np.zeros((n_samples, offdProd.shape[0], i))
        for j in range(i):
            x_comb[:, :, j] = X[:, offdProd[:, j]]
        X_allpairs = np.append(X_allpairs, np.prod(x_comb, axis=2), axis=1)

    return X_allpairs
