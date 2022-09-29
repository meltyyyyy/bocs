import numpy as np
import numpy.typing as npt


def sample_binary_matrix(n_samples: int, n_vars: int) -> npt.NDArray:
    """Sample binary matrix

    Args:
        n_samples (int): The number of samples.
        n_vars (int): The number of variables.

    Returns:
        np.ndarray: Binary matrix of shape (n_samples, n_vars)
    """
    # Generate matrix of zeros with ones along diagonals
    sample = np.zeros((n_samples, n_vars))

    # Sample model indices
    sample_num = np.random.randint(2**n_vars, size=n_samples)

    strformat = '{0:0' + str(n_vars) + 'b}'
    # Construct each binary model vector
    for i in range(n_samples):
        model = strformat.format(sample_num[i])
        sample[i, :] = np.array([int(b) for b in model])

    return sample


def sample_integer_matrix(n_samples: int, low: int, high: int, n_vars: int) -> npt.NDArray:
    """Sample Integer matrix

    Args:
        n_samples (int): The number of samples.
        min (int): Minimum of integer.
        max (int): Maximum of integer.

    Returns:
        npt.NDArray: Interger matrix of shape (n_samples, max - min)
    """
    sample = np.zeros((n_samples, n_vars))
    for i in range(n_samples):
        sample[i, :] = np.random.randint(low, high + 1, size=n_vars)

    return sample
