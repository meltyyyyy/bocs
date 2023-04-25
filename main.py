import numpy as np

exp = 'miqp'

for i in [8, 12, 16]:
    dirname = f'/root/bocs/runs/ohe/milp/blr/{i}/'

    data = []
    for j in range(100):
        try:
            data.append(np.load(dirname + f'{j}_03.npy'))
        except Exception as e:
            print(e)
    data = np.array(data).T
    print(data.shape)

    # mean = np.mean(data, axis=1)
    # norm = mean[0]
    # mean = mean / norm
    # std_err = np.std(data, axis=1, ddof=1) / norm / np.sqrt(data.shape[1])
    # data = np.stack([mean, std_err], axis=1)

    # np.savetxt(f'normalized_{i}.txt', data)
