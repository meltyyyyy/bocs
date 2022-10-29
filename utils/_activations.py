import numpy as np
import numpy.typing as npt


def relu(x: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, x)
