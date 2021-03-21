from numerical_study.ns_utils import Generate_d_rs, Calculate_Sampled_Sigma_Matrix, Calculate_Sampled_Cov_Matrix
from numerical_study.ns_utils import Generate_Graph
import numpy as np
m, n, = 6, 20
f, h, = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.12, 0.11, 0.10, 0.09, 0.11, 0.10]
samples = Generate_d_rs([20, 25, 30, 25] * 5, 0.5, 0.5, 15, dist=1, seed=0)
mu_sampled = samples.mean(axis=0)
sigma_sampled = Calculate_Sampled_Sigma_Matrix(samples)
sigma_sampled = sigma_sampled + np.diag(np.ones(n))*0.1
cov_sampled = Calculate_Sampled_Cov_Matrix(samples)
graph = Generate_Graph(m, n, int((m+n)*2), 0)
mu_lb = mu_sampled * 0.8
mu_ub = mu_sampled * 1.2
sigma_lb = sigma_sampled * 0.8
sigma_ub = sigma_sampled * 1.2