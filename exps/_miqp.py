import numpy as np
import numpy.typing as npt


def miqp(n_vars: int) -> npt.NDArray:
    Q = np.random.randn(n_vars, n_vars)
    Q = (Q + Q.T) / 2
    return Q
