import numpy as np
import numpy.typing as npt


def fitting_curve(t: npt.NDArray, n_vars: int, alpha: float = 10.5, beta: float = 0.08):
    tau = n_vars ** alpha
    alg_idx, exp_idx = t < tau, tau <= t
    y = np.zeros_like(t).astype(float)
    y[alg_idx] = t[alg_idx] ** (-beta)
    y[exp_idx] = (tau ** (-beta)) * np.exp(0.30 * (1 - (t[exp_idx] / tau)))
    return y
