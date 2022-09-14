import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


# Bayesian Horseshoe Sampler
# References:
# A simple sampler for the horseshoe estimator
# https://arxiv.org/pdf/1508.03884.pdf
def bhs(X, y, n_samples=1000, burnin=200):
    n, p = X.shape
    XtX = X.T @ X

    beta = np.zeros((p, n_samples))
    sigma2 = 1
    lambda2 = np.random.uniform(size=p)
    tau2 = 1
    nu = np.ones(p)
    xi = 1

    # Run Gibbs Sampler
    for i in range(n_samples):
        Lambda_star = tau2 * np.diag(lambda2)
        A = XtX + np.linalg.inv(Lambda_star)
        A_inv = np.linalg.inv(A)
        b = np.random.multivariate_normal(A_inv @ X.T @ y, sigma2 * A_inv)

        # Sample sigma^2
        e = y - np.dot(X, b)
        shape = (n + p) / 2.
        scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
        sigma2 = 1. / np.random.gamma(shape, 1. / scale)

        # Sample lambda^2
        scale = 1. / nu + b**2. / 2. / tau2 / sigma2
        lambda2 = 1. / np.random.exponential(1. / scale)

        # Sample tau^2
        shape = (p + 1.) / 2.
        scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
        tau2 = 1. / np.random.gamma(shape, 1. / scale)

        # Sample nu
        scale = 1. + 1. / lambda2
        nu = 1. / np.random.exponential(1. / scale)

        # Sample xi
        scale = 1. + 1. / tau2
        xi = 1. / np.random.exponential(1. / scale)

        if iter > burnin:
            beta[:, i] = b

    return beta


if __name__ == '__main__':
