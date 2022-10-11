import numpy as np
import matplotlib.pylab as plt
plt.style.use('seaborn-pastel')

result_ohe = np.load('sa_ohe.npy')
result_be = np.load('sa_be.npy')

n_iter = np.arange(result_be.shape[0])
true_opt = 36

mean_ohe = np.abs(np.mean(result_ohe, axis=1) - true_opt)
mean_be = np.abs(np.mean(result_be, axis=1) - true_opt)
var_ohe = np.var(result_ohe, axis=1)
var_be = np.var(result_be, axis=1)
std_ohe = np.sqrt(np.abs(var_ohe))
std_be = np.sqrt(np.abs(var_be))

fig = plt.figure(figsize=(12, 8))
plt.yscale('log')
plt.xlabel('Iteration ' + r'$t$', fontsize=18)
plt.ylabel(r'$|f(x_t)-f(x^*)|$', fontsize=18)
plt.plot(n_iter, mean_ohe, label='One Hot Encoding')
# plt.fill_between(n_iter, mean_ohe + 2 * std_ohe, mean_ohe - 2 * std_ohe, alpha=.2, label="95% Confidence Interval")
plt.plot(n_iter, mean_be, label='Binary Expansion')
plt.fill_between(n_iter, mean_be + 2 * std_be, mean_be - 2 * std_be, alpha=.2, label="95% Confidence Interval")
plt.legend()
fig.savefig('figs/bocs.png')
plt.close(fig)
