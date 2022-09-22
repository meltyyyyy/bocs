import numpy as np

rs = np.random.RandomState(42)

def sample_binary_matrix(n_samples: np.int64, n_vars: np.int64) -> np.ndarray:
    """Sample binary matrix

    Args:
        n_samples (np.int64): The number of samples.
        n_vars (np.int64): The number of variables.

    Returns:
        np.ndarray: Binary matrix of shape (n_samples, n_vars)
    """
    # Generate matrix of zeros with ones along diagonals
    sample = np.zeros((n_samples, n_vars))

    # Sample model indices
    sample_num = rs.randint(2**n_vars, size=n_samples)

    strformat = '{0:0' + str(n_vars) + 'b}'
    # Construct each binary model vector
    for i in range(n_samples):
        model = strformat.format(sample_num[i])
        sample[i, :] = np.array([int(b) for b in model])

    return sample
