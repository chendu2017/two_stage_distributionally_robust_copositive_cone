from numerical_study.ns_utils import Generate_d_rs, Calculate_Sampled_Sigma_Matrix, Calculate_Sampled_Cov_Matrix
m, n, = 2, 3
f, h, = [0, 5], [0.1, 0.11]
samples = Generate_d_rs([20, 25, 30], 0.5, 0.5, 15, dist=1, seed=0)
mu_sampled = samples.mean(axis=0)
sigma_sampled = Calculate_Sampled_Sigma_Matrix(samples)
cov_sampled = Calculate_Sampled_Cov_Matrix(samples)
graph = [[1, 0, 1],
         [1, 1, 1]]
mu_lb = mu_sampled * 0.8
mu_ub = mu_sampled * 1.2
sigma_lb = sigma_sampled * 0.8
sigma_ub = sigma_sampled * 1.2