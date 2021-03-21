from numerical_study.ns_utils import Generate_d_rs, Calculate_Sampled_Sigma_Matrix, Calculate_Sampled_Cov_Matrix
from numerical_study.ns_utils import Generate_Graph
m, n, = 4, 8
f, h, = [20, 30, 40, 50], [0.13, 0.12, 0.11, 0.10]
samples = Generate_d_rs([200, 250, 300, 250, 200, 150, 300, 350], 0.5, 0.5, 15, dist=1, seed=0)
mu_sampled = samples.mean(axis=0)
sigma_sampled = Calculate_Sampled_Sigma_Matrix(samples)
cov_sampled = Calculate_Sampled_Cov_Matrix(samples)
graph = Generate_Graph(m, n, int((m+n)*2), 0)
mu_lb = mu_sampled * 0.8
mu_ub = mu_sampled * 1.2
sigma_lb = sigma_sampled * 0.8
sigma_ub = sigma_sampled * 1.2