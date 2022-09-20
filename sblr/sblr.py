import numpy as np


class SparseBayesianLinearRegression:
    def __init__(self, n_vars: int, order: int, random_state: int = 42):
        self.n_vars = n_vars
        self.order = order
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Sparse Bayesian Linear Regression

        Args:
            X (np.ndarray): matrix of shape (n_samples, n_vars)
            y (np.ndarray): matrix of shape (n_samples, )
        """
        assert X.shape[1] != self.n_vars, "The number of variables does not match."

        

