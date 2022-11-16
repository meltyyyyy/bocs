import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_config

config = get_config()


def plot_bocs(filepath: str):
    dirname = os.path.dirname(filepath)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    encode = 'One Hot Encode' if filename.split("_")[0] == 'ohe' else "Binary Expansion"
    n_vars = filename.split("_")[1]
    data = np.load(f'{filepath}')

    n_iter = np.arange(data.shape[0])
    mean = np.mean(data, axis=1)
    var = np.var(data, axis=1)
    std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

    fig = plt.figure(figsize=(12, 8))
    plt.title(f'MILP with {n_vars} variables')
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")
    plt.plot(n_iter, mean, label=f'BOCS + {encode}')
    plt.fill_between(n_iter, mean + 2 * std_err, mean - 2 * std_err, alpha=.2)
    plt.legend()
    filepath = dirname + '/' + filename + '.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)
