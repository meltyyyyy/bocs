import numpy as np

def fast_mvgs(Phi : np.ndarray, alpha : np.ndarray, D : np.ndarray) -> np.ndarray:
    """Fast sampler for Multivariate Gaussian distributions
    Applicable for large p, p > n of
    the form N(mu, S), where
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
    v = np.dot(Phi,u) + delta
    #w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
    #x = u + np.dot(D,np.dot(Phi.T,w))
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x
