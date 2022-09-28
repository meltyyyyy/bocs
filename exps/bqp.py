import numpy as np
import numpy.typing as npt


def bqp(n_vars: int, alpha: int) -> npt.NDArray:
    i = np.linspace(1, n_vars, n_vars)
    j = np.linspace(1, n_vars, n_vars)

    def K(s, t): return np.exp(-1 * (s - t)**2 / alpha)
    decay = K(i[:, None], j[None, :])

    Q = np.random.randn(n_vars, n_vars)
    Q = Q * decay

    return Q
