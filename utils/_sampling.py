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

    for i in range(n_samples):
        n_bit = n_vars
        x = []

        while n_bit > 0:
            # Sample model indices
            sample_num = np.random.randint(2**n_bit if n_bit < 32 else 32)
            strformat = '{0:0' + str(n_bit) + 'b}' if n_bit < 32 else '{0:0' + str(32) + 'b}'
            model = strformat.format(sample_num)
            x = x + [int(b) for b in model]
            n_bit -= 32
        sample[i, :] = np.array(x)

    return sample


def sample_integer_matrix(n_samples: int, low: int, high: int, n_vars: int) -> npt.NDArray:
    """Sample Integer matrix

    Args:
        n_samples (int): The number of samples.
        low (int): Minimum of integer.
        high (int): Maximum of integer.
        n)vars (int) : The number of variables.

    Returns:
        npt.NDArray: Interger matrix of shape (n_samples, max - min)
    """
    sample = np.zeros((n_samples, n_vars))
    for i in range(n_samples):
        sample[i, :] = np.random.randint(low, high + 1, size=n_vars)

    return sample


if __name__ == "__main__":
    sample_binary_matrix(10, 100)
