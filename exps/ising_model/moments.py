import numpy as np
import numpy.typing as npt


def moments(Q: npt.NDArray) -> npt.NDArray:
    n_vars = Q.shape[0]

    # generate all binary vectors
    vectorizer = np.vectorize(np.binary_repr)
    bins = vectorizer(np.arange(0, 2 ** n_vars), width=n_vars)
    bins = list(map(lambda b: list(map(int, list(b))), bins))
    bins = np.array(bins)
    bins = np.where(bins == 0, -1, bins)

    n_vec = bins.shape[0]
    pdfs = np.zeros(n_vec)
    for i in range(n_vec):
        pdfs[i] = np.exp(bins[i, :] @ Q @ bins[i, :].T)

    # compute normalizing constant
    norm = np.sum(pdfs)

    moment = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            bin_pair = bins[:, i] * bins[:, j]
            moment[i, j] = np.sum(bin_pair * pdfs) / norm

    return moment
