import numpy as np
import matplotlib.pylab as plt
plt.style.use('seaborn-v0_8-pastel')

result_ohe = np.load('ohe_time.npy')
result_be = np.load('be_time.npy')

n_vars = np.arange(result_be.shape[0])

mean_ohe = np.mean(result_ohe, axis=1)
mean_be = np.mean(result_be, axis=1)
var_ohe = np.var(result_ohe, axis=1)
var_be = np.var(result_be, axis=1)
std_ohe = np.sqrt(np.abs(var_ohe))
std_be = np.sqrt(np.abs(var_be))

fig = plt.figure(figsize=(12, 8))
plt.yscale('linear')
plt.xlabel('Number of variables', fontsize=18)
plt.ylabel('Time', fontsize=18)
plt.plot(n_vars, mean_ohe, label='One Hot Encoding')
plt.fill_between(n_vars, mean_ohe + 2 * std_ohe, mean_ohe -
                 2 * std_ohe, alpha=.2, label="95% Confidence Interval")
plt.plot(n_vars, mean_be, label='Binary Expansion')
plt.fill_between(n_vars, mean_be + 2 * std_be, mean_be - 2 *
                 std_be, alpha=.2, label="95% Confidence Interval")
plt.legend()
fig.savefig('figs/bocs.png')
plt.close(fig)
