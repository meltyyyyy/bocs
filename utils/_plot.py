import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from utils import get_config
from ._fitting_curve import fitting_curve
import glob

plt.style.use('seaborn-v0_8-pastel')
config = get_config()


def plot_bocs(filepath: str):
    dirname = os.path.dirname(filepath)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    encode = 'One Hot Encode' if filename.split(
        "_")[0] == 'ohe' else "Binary Expansion"
    n_vars = filename.split("_")[1]
    data = np.load(f'{filepath}')

    n_iter = np.arange(data.shape[0])
    mean = np.mean(data, axis=1)
    var = np.var(data, axis=1)
    std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

    fig = plt.figure(figsize=(12, 8))
    plt.title(f'MILP with {n_vars} variables')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(0, linestyle="dashed")
    plt.plot(n_iter, mean, label=f'BOCS + {encode}')
    plt.fill_between(n_iter, mean + 2 * std_err, mean - 2 * std_err, alpha=.5)
    plt.legend()
    filepath = dirname + '/' + filename + '.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


def plot_time_dependency(exp: str):
    dirname = config['output_dir'] + exp + '/time/'
    def key(x): return int(os.path.splitext(
        os.path.basename(x))[0].split('_')[1])
    filepaths = sorted(glob.glob(dirname + "ohe_*.npy"), key=key)

    fig = plt.figure(figsize=(12, 8))
    plt.title('MILP with BOCS')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")

    for filepath in filepaths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        n_vars = filename.split("_")[1]
        data = np.load(f'{filepath}')

        # standalize
        for i in range(data.shape[1]):
            data[:, i] = 1e-7 + data[:, i] / (data[0, i] + 10e-7)
        n_iter = np.arange(data.shape[0]) + 1
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

        plt.plot(n_iter, mean, label=f'n_vars : {n_vars}')
        plt.fill_between(n_iter, mean + 2 * std_err,
                         mean - 2 * std_err, alpha=.5)

    plt.legend()
    filepath = dirname + '/' + 'time_dependency.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


def plot_range_dependency(exp: str):
    dirname = config['output_dir'] + exp + '/range/'
    def key(x): return int(os.path.splitext(
        os.path.basename(x))[0].split('_')[2])
    filepaths = sorted(glob.glob(dirname + "ohe_*.npy"), key=key)

    fig = plt.figure(figsize=(12, 8))
    plt.title('3 variable MILP with BOCS')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")

    for i in range(26, len(filepaths), 1):
        filepath = filepaths[i]
        filename = os.path.splitext(os.path.basename(filepath))[0]
        var_range = filename.split("_")[2]
        data = np.load(f'{filepath}')

        # standalize
        for i in range(data.shape[1]):
            data[:, i] = 10e-7 + data[:, i] / (data[0, i] + 10e-7)

        n_iter = np.arange(data.shape[0]) + 1
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

        plt.plot(n_iter, mean, label=f'range : 0 ~ {var_range[1:]}')
        plt.fill_between(n_iter, mean + 2 * std_err,
                         mean - 2 * std_err, alpha=.5)

    plt.legend()
    filepath = dirname + '/' + 'range_dependency.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)


def plot_fitting_curve(filepaths: List[str]):
    fig = plt.figure(figsize=(12, 8))
    plt.title('Fitting curve')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.axhline(0, linestyle="dashed")

    for filepath in filepaths:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        n_vars = int(filename.split("_")[1])
        data = np.load(f'{filepath}')
        print(data.shape)

        # standalize
        for i in range(data.shape[1]):
            data[:, i] = 10e-7 + data[:, i] / (data[0, i] + 10e-7)

        n_iter = np.arange(data.shape[0]) + 1
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1)
        std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])

        # for data
        plt.plot(n_iter, mean, label=f'n_vars : {n_vars}')
        plt.fill_between(n_iter, mean + 2 * std_err,
                         mean - 2 * std_err, alpha=.5)

        # for fitting
        plt.plot(
            n_iter,
            fitting_curve(n_iter, n_vars=n_vars),
            linestyle='dashed',
            label=f'fitting curve for {n_vars} vars')

    plt.legend()
    filepath = config['output_dir'] + 'fitting_curve.png'
    fig.savefig(f'{filepath}')
    plt.close(fig)
