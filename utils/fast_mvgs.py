import numpy as np


def fast_mvgs(Phi: np.ndarray, alpha: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Fast sampler for Multivariate Gaussian distributions
    Applicable for large p, p > n of
    the form N(mu, Σ), where
        mu = Σ Phi^T y
        Σ  = (Phi^T Phi + D^-1)^-1

    Time complexity is O(n^2 p)

    <Reference>
    Fast sampling with Gaussian scale-mixture priors in high-dimensional regression.
    https://arxiv.org/pdf/1506.04778.pdf

    Args:
        Phi (np.ndarray): Matrix of shape (n, p)
        alpha (np.ndarray): Array of shape (n, 1)
        D (np.ndarray): Matrix of shape (p, p)

    Returns:
        np.ndarray: Array os shape (p, 1)
    """

    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi, u) + delta
    #w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
    #x = u + np.dot(D,np.dot(Phi.T,w))
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:, np.newaxis])
    w = np.linalg.solve(np.matmul(Phi, Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt, w)

    return x


def fast_mvgs_(Phi: np.ndarray, PtP: np.ndarray, alpha: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Fast sampler for Multivariate Gaussian distributions

    Applicable for small p of
    the form N(mu, Σ), where
        mu = Σ Phi' y
        Σ  = (Phi^T Phi + D^-1)^-1

    Time complexity is O(n)

    <Reference>
    Fast sampling of gaussian markov random fields.
    https://arxiv.org/pdf/1506.04778.pdf

    Args:
        Phi (np.ndarray): Matrix of shape (n, p)
        PtP (np.ndarray): Matrix of shape (p, p)
        alpha (np.ndarray): Array of shape (n, 1)
        D (np.ndarray): Matrix of shape (p, p)
    """

    p = Phi.shape[1]
    D_inv = np.diag(1. / np.diag(D))

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = np.linalg.cholesky(PtP + D_inv)
    except BaseException:
        M = PtP + D_inv
        M = (M + M.T) / 2.
        max_eig = np.max(np.linalg.eigvals(M))
        max_eig = np.real_if_close(max_eig)
        L = np.linalg.cholesky(M + max_eig * 1e-15 * np.eye(M.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T, alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))

    x = m + w

    return x
