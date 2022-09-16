import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def horseshoe(tau=1.0):
    k = np.random.beta(0.5, 0.5, size=100)
    λ = np.sqrt(-1 + 1 / k)
    scale = (tau ** 2) * (λ ** 2)
    samples = np.random.normal(0, scale)
    return samples


def plot(samples):
    fig = plt.figure()
    plt.hist(samples, bins=100)
    plt.savefig('horseshoe.png')
    plt.close()


def main():
    beta = np.random.beta(0.5, 0.5, size=1000000)
    fig = plt.figure()
    plt.hist(beta, bins=100)
    plt.savefig('dist-k.png')
    plt.close()

    samples = horseshoe()
    fig = plt.figure()
    plt.hist(samples, bins=100)
    plt.savefig('horseshoe.png')
    plt.close()




if __name__ == "__main__":
    main()
