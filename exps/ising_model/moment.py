import numpy as np
import numpy.typing as npt


def moment(Q: npt.NDArray, width: int = 8) -> npt.NDArray:
    n_vars = Q.shape[0]
    vectorizer = np.vectorize(np.binary_repr)
    bins = vectorizer(np.arange(0, 2 ** n_vars), width=width)
    
