import numpy as np
import numpy.typing as npt
import cvxpy as cvx
from itertools import combinations


def sdp_relaxation(alpha: npt.NDArray, n_vars: int):

    # Extract vector of coefficients
    b = alpha[1:n_vars + 1]
    a = alpha[n_vars + 1:]

    # get indices for quadratic terms
    idx_prod = np.array(list(combinations(np.arange(n_vars), 2)))
    n_idx = idx_prod.shape[0]

    # check number of coefficients
    if a.size != n_idx:
        raise ValueError('Number of Coefficients does not match indices!')

    # Convert a to matrix form
    A = np.zeros((n_vars, n_vars))
    for i in range(n_idx):
        A[idx_prod[i, 0], idx_prod[i, 1]] = a[i] / 2.
        A[idx_prod[i, 1], idx_prod[i, 0]] = a[i] / 2.

    # Convert to standard form
    bt = b / 2. + np.dot(A, np.ones(n_vars)) / 2.
    bt = bt.reshape((n_vars, 1))
    At = np.vstack((np.append(A / 4., bt / 2., axis=1), np.append(bt.T, 2.)))

    # Run SDP relaxation
    X = cvx.Variable((n_vars + 1, n_vars + 1), PSD=True)
    obj = cvx.Minimize(cvx.trace(cvx.matmul(At, X)))
    constraints = [cvx.diag(X) == np.ones(n_vars + 1)]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    # Extract vectors and compute Cholesky
    # add small identity matrix is X.value is numerically not PSD
    try:
        L = np.linalg.cholesky(X.value)
    except BaseException:
        XpI = X.value + 1e-15 * np.eye(n_vars + 1)
        L = np.linalg.cholesky(XpI)

    # Repeat rounding for different vectors
    n_rand_vector = 100

    X_vect = np.zeros((n_vars, n_rand_vector))
    obj_vect = np.zeros(n_rand_vector)

    for kk in range(n_rand_vector):

        # Generate a random cutting plane vector (uniformly
        # distributed on the unit sphere - normalized vector)
        r = np.random.randn(n_vars + 1)
        r = r / np.linalg.norm(r)
        y_soln = np.sign(np.dot(L.T, r))

        # convert solution to original domain and assign to output vector
        X_vect[:, kk] = (y_soln[:n_vars] + 1.) / 2.
        obj_vect[kk] = np.dot(np.dot(X_vect[:, kk].T, A), X_vect[:, kk]) \
            + np.dot(b, X_vect[:, kk])

    # Find optimal rounded solution
    opt_idx = np.argmin(obj_vect)
    model = X_vect[:, opt_idx]
    obj = obj_vect[opt_idx]

    return (model, obj)
