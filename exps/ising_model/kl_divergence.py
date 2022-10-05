import numpy as np
import numpy.typing as npt


def kl_divergence(Theta_P: npt.NDArray, moments: npt.NDArray, x: npt.NDArray, width: int = 8):
    n_vars = Theta_P.shape[0]

    # generate all binary vectors
    vectorizer = np.vectorize(np.binary_repr)
    bins = vectorizer(np.arange(0, 2 ** n_vars), width=width)
    bins = list(map(lambda b: list(map(int, list(b))), bins))
    bins = np.array(bins)
    bins = np.where(bins == 0, -1, bins)

    n_vec = bins.shape[0]
    Ps = np.zeros(n_vec)
    for i in range(n_vec):
        Ps[i] = np.exp(bins[i, :] @ Theta_P @ bins[i, :].T)
    Zp = np.sum(Ps)

    len_x = x.shape[0]
    KL = np.zeros(len_x)

    for i in range(len_x):
        Theta_Q = np.tril(Theta_P, -1)
        nnz_Q = np.nonzero(Theta_Q)
        Theta_Q[nnz_Q] = Theta_Q[nnz_Q] @ x[i, :]
        Theta_Q = Theta_Q + Theta_Q.T

        Qs = np.zeros(n_vec)
        for j in range(n_vec):
            Qs[j] = np.exp(bins[j, :] @ Theta_Q @ bins[j, :].T)
        Zq = np.sum(Qs)

        KL[j] = np.sum(np.sum((Theta_P - Theta_Q) @ moments)) + np.log(Zq) - np.log(Zp)

    return KL
