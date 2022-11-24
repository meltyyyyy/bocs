from utils import plot_range_dependency, plot_time_dependency, get_config, plot_bocs

exp = 'milp'
# plot_time_dependency(exp)
# plot_range_dependency(exp)
config = get_config()

# for i in range(3, 23, 1):
#     filepath = config['output_dir'] + f'milp/time/ohe_{i}.npy'
#     plot_bocs(filepath)
# filepath = config['output_dir'] + 'annealings/sa/5_03.npy'
# plot_bocs(filepath)
filepath = config['output_dir'] + 'annealings/sqa/5_03.npy'
plot_bocs(filepath)

import numpy as np
data = np.load(filepath)
print(data.shape)
