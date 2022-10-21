
from exps import load_study
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
from surrogates import SparseBayesianLinearRegression
from aquisitions import simulated_annealing
from utils import sample_integer_matrix, encode_binary, decode_binary
from log import get_logger

logger = get_logger(__name__, __file__)
STUDY_DIR = '/root/bocs/study/'


def bocs_sa_be(objective, low: int, high: int, n_vars: int, n_init: int = 10,
               n_trial: int = 100, sa_reruns: int = 5):
    # Set the number of Simulated Annealing reruns
    sa_reruns = 5

    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Binary expansion
    n_bit = len(bin(high)[2:])
    X = encode_binary(high, n_vars, X)

    # Define surrogate model
    sblr = SparseBayesianLinearRegression(n_bit * n_vars, 1)
    sblr.fit(X, y)

    for _ in range(n_trial):

        def surrogate_model(x): return sblr.predict(x)

        sa_X = np.zeros((sa_reruns, n_bit * n_vars))
        sa_y = np.zeros(sa_reruns)

        for j in range(sa_reruns):
            opt_X, opt_y = simulated_annealing(
                surrogate_model, n_bit * n_vars, n_iter=200)
            sa_X[j, :] = opt_X[-1, :]
            sa_y[j] = opt_y[-1]

        max_idx = np.argmax(sa_y)
        x_new = sa_X[max_idx, :]

        # evaluate model objective at new evaluation point
        x_new = x_new.reshape((1, n_bit * n_vars))
        y_new = objective(decode_binary(high, n_vars, x_new))

        # Update posterior
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        sblr.fit(X, y)

        # log current solution
        x_curr = decode_binary(high, n_vars, x_new).astype(int)
        y_curr = objective(x_curr)
        logger.info(f'x{i}: {x_curr[0]}, y{i}: {y_curr[0]}')

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def plot(result: npt.NDArray, true_opt: float):
    n_iter = np.arange(result.shape[0])
    mean = np.abs(np.mean(result, axis=1) - true_opt)
    var = np.var(result, axis=1)
    std = np.sqrt(np.abs(var))

    fig = plt.figure(figsize=(12, 8))
    plt.yscale('linear')
    plt.ylim((10e-4, 100))
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x_t)-f(x^*)|$', fontsize=18)
    plt.plot(n_iter, mean)
    plt.fill_between(n_iter, mean + 2 * std, mean - 2 * std, alpha=.2)
    fig.savefig('figs/bocs/sa_be_10.png')
    plt.close(fig)


if __name__ == "__main__":
    n_vars = 5
    experiment = 'bqp'
    study = load_study(experiment, f'{n_vars}.json')
    Q = study['Q']
    n_runs = study['n_runs']
    n_runs = 2

    def objective(X: npt.NDArray) -> npt.NDArray:
        return - np.diag(X @ Q @ X.T)

    # Run Bayesian Optimization
    n_trial = 100
    result = np.zeros((n_trial, n_runs))

    for i in range(n_runs):
        X, y = bocs_sa_be(objective,
                          low=0,
                          high=9,
                          n_trial=n_trial,
                          n_vars=n_vars)
        y = np.maximum.accumulate(y)
        result[:, i] = y
        logger.info('best_y: {}'.format(y[-1]))

    np.save(STUDY_DIR + experiment + '/' + f'{n_vars}.npy', result)
    plot(result, 0)
