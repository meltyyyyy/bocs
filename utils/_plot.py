import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_config
import glob
plt.style.use('seaborn-v0_8-pastel')

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


def plot_time_dependency(exp: str):
    dirname = config['output_dir'] + exp + '/time/'
    def key(x): return int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    filepaths = sorted(glob.glob(dirname + "*.npy"), key=key)

    fig = plt.figure(figsize=(12, 8))
    plt.title('MILP with BOCS')
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")

    for filepath in filepaths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        n_vars = filename.split("_")[1]
        data = np.load(f'{filepath}')

        # standalize
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] / (data[0, i] + 1e-7)
        n_iter = np.arange(data.shape[0])
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

        plt.plot(n_iter, mean, label=f'n_vars : {n_vars}')
        plt.fill_between(n_iter, mean + 2 * std_err, mean - 2 * std_err, alpha=.2)

    plt.legend()
    filepath = dirname + '/' + 'time_dependency.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


def plot_range_dependency(exp: str):
    dirname = config['output_dir'] + exp + '/range/'
    def key(x): return int(os.path.splitext(os.path.basename(x))[0].split('_')[2])
    filepaths = sorted(glob.glob(dirname + "*.npy"), key=key)

    fig = plt.figure(figsize=(12, 8))
    plt.title('3 variable MILP with BOCS')
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")

    for i in range(0, len(filepaths), 4):
        filepath = filepaths[i]
        filename = os.path.splitext(os.path.basename(filepath))[0]
        var_range = filename.split("_")[2]
        data = np.load(f'{filepath}')

        # standalize
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] / (data[0, i] + 1e-7)

        n_iter = np.arange(data.shape[0])
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

        plt.plot(n_iter, mean, label=f'range : 0 ~ {var_range[1:]}')
        plt.fill_between(n_iter, mean + 2 * std_err, mean - 2 * std_err, alpha=.2)

    plt.legend()
    filepath = dirname + '/' + 'range_dependency.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)
