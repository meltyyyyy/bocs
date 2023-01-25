from utils import get_config
import numpy as np


exp = 'miqp'
config = get_config()

for i in [5, 6, 7, 8, 9]:
    dirname = config['output_dir'] + f'annealings/sqa/miqp/dwave/{i}/'

    data = []
    for j in range(100):
        try:
            data.append(np.load(dirname + f'{j}_03.npy'))
        except:
            print("error")
    data = np.array(data).T
    print(data.shape)

    mean = np.mean(data, axis=1)
    norm = mean[0]
    mean = mean / norm
    std_err = np.std(data, axis=1, ddof=1) / norm / np.sqrt(data.shape[1])
    data = np.stack([mean, std_err], axis=1)

    np.savetxt(f'normalized_{i}.txt', data)
