import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_and_save(result: npt.NDArray, n_vars: int):
    n_iter = np.arange(result.shape[0])
    mean = np.mean(result, axis=1)
    var = np.var(result, axis=1)
    std = np.sqrt(np.abs(var))

    fig = plt.figure(figsize=(12, 8))
    plt.title(f'MILP with {n_vars} variables')
    plt.yscale('linear')
    plt.xlabel('Iteration ' + r'$t$', fontsize=18)
    plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
    plt.plot(n_iter, mean, label='BOCS + One Hot Encode')
    plt.fill_between(n_iter, mean + 2 * std, mean - 2 * std, alpha=.2)
    plt.legend()

    # save
    now = datetime.now()
    filedir = config['output_dir'] + f'{EXP}/' + now.strftime("%m%d") + '/'

    fig.savefig(f'{filedir}' + f'{EXP}_ohe_{n_vars}.png')
    np.save(f'{filedir}' + f'{EXP}_ohe_{n_vars}.npy', result)
    plt.close(fig)
