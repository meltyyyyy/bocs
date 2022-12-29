from utils import get_config
import numpy as np


exp = 'milp'
config = get_config()
dirname = config['output_dir'] + 'milp/time/ohe_sblr_'


for i in [6, 7, 8, 9]:
    data = np.load(dirname + f'{i}.npy')
    print(data.shape)

    mean = np.mean(data, axis=1)
    norm = mean[0]
    mean = mean / norm
    std_err = np.std(data, axis=1, ddof=1) / norm / np.sqrt(data.shape[1])
    data = np.stack([mean, std_err], axis=1)

    np.savetxt(f'normalized_{i}.txt', data)
