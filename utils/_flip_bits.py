import numpy as np
import numpy.typing as npt


def flip_bits(x: npt.NDArray, n_flips: int) -> npt.NDArray:
    assert x.shape[0] == 1, "Only 2 dimensional array with 1 row can be flipped."
    assert x.ndim == 2, "Dimension of x must be 2."

    n_vars = x.shape[1]
    for _ in range(n_flips):
        flip_bit = np.random.randint(n_vars)
        x[0, flip_bit] = 1 - x[0, flip_bit]

    return x
