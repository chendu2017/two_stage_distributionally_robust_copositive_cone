from numpy.random import choice
import numpy as np
from time import time


def Bootstrap(samples, CI=95, replicates=100000, seed=int(time())):
    samples = np.asarray(samples)
    N, n = samples.shape   # N samples, n locations

    bootstrapped_means = []
    bootstrapped_sigmas = []

    np.random.seed(seed)
    for r in range(replicates):
        chosen_samples = samples[choice(N, size=N, replace=True)]
        bootstrapped_means.append(chosen_samples.mean(axis=0))
        bootstrapped_sigmas.append(np.matmul(chosen_samples.T, chosen_samples) / N)

    sample_mean_lb, sample_mean_ub = np.percentile(bootstrapped_means, q=[(100-CI) / 2, 100 - (100-CI) / 2], axis=0)
    sample_sigma_lb, sample_sigma_ub = np.percentile(bootstrapped_sigmas, q=[(100-CI) / 2, 100 - (100-CI) / 2], axis=0)
    return sample_mean_lb, sample_mean_ub, sample_sigma_lb, sample_sigma_ub















