import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def horseshoe(tau=1.0):
    k = np.random.beta(0.5, 0.5, size=10000)
    λ = np.sqrt(-1 + 1 / k)
    scale = (tau ** 2) * (λ ** 2)
    samples = np.random.normal(0, scale)
    return samples


def main():
    beta = np.random.beta(0.3, 0.3, size=100000)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Beta ditribution", fontsize=18, fontweight='bold')
    axes[0].set_title("Beta(0.3, 0.3)")
    axes[0].hist(beta, bins=100)

    beta = np.random.beta(0.5, 0.5, size=100000)
    axes[1].set_title("Beta(0.5, 0.5)")
    axes[1].hist(beta, bins=100)

    beta = np.random.beta(0.7, 0.7, size=100000)
    axes[2].set_title("Beta(0.7, 0.7)")
    axes[2].hist(beta, bins=100)

    fig.tight_layout()
    fig.savefig("dist-k.png")
    plt.close()

    samples = horseshoe(tau=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Horseshoe ditribution", fontsize=18, fontweight='bold')
    axes[0].set_title("tau=1.0")
    axes[0].hist(samples, bins=100)

    samples = horseshoe(tau=100.0)
    axes[1].set_title("tau=100")
    axes[1].hist(samples, bins=100)

    samples = horseshoe(tau=0.01)
    axes[1].set_title("tau=0.01")
    axes[1].hist(samples, bins=100)
    plt.close()




if __name__ == "__main__":
    main()
