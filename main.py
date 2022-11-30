from utils import get_config
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
plt.style.use("seaborn-v0_8-colorblind")

exp = 'milp'
config = get_config()
dirname = config['output_dir'] + exp + '/time/'

fig = plt.figure(figsize=(12, 8))
plt.title('MILP with BOCS')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Iteration ' + r'$t$', fontsize=18)
plt.ylabel(r'$|f(x) - f(x_t)|$', fontsize=18)
plt.axhline(0, linestyle="dashed")


def q(t: npt.NDArray, n_vars: int, alpha: float, beta: float = 0.25):
    tau = n_vars ** alpha
    alg_idx, exp_idx = t < tau, tau <= t
    y = np.zeros_like(t).astype(float)
    y[alg_idx] = t[alg_idx] ** (-beta)
    y[exp_idx] = (tau ** (-beta)) * np.exp(0.20 * (1 - (t[exp_idx] / tau)))
    return y


for i in range(5, 11):
    filepath = dirname + f'ohe_{i}.npy'
    filename = os.path.splitext(os.path.basename(filepath))[0]
    n_vars = filename.split("_")[1]
    data = np.load(f'{filepath}')
    for i in range(data.shape[1]):
        data[:, i] = 10e-7 + data[:, i] / (data[0, i] + 10e-7)
    n_iter = np.arange(data.shape[0])
    mean = np.mean(data, axis=1)
    var = np.var(data, axis=1)
    std_err = np.sqrt(np.abs(var)) / np.sqrt(data.shape[0])
    plt.plot(n_iter + 1, mean, label=f'n_vars : {n_vars}', )
    plt.fill_between(n_iter + 1, mean + 2 * std_err, mean - 2 * std_err, alpha=.5)

# for fitting
plt.plot(n_iter + 1, q(n_iter + 1, n_vars=5, alpha=2.15), label='fitting curve for 5 vars')

plt.legend()
fig.savefig('temp.png')
plt.close(fig)
