from copy import deepcopy

import numpy as np

from numerical_study.ns_utils import Calculate_Cov_Matrix, Generate_d_rs

# e_param
m, n = 4, 4
f, h = [5, 5, 5, 5], [0.1, 0.1, 0.1, 0.1]
kappa = 1.0
cv = 0.2
rho = 0.2
graph = [[1, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 0, 1, 1]]
mu = [20, 20, 20, 20]
cov_matrix = Calculate_Cov_Matrix(mu, cv, rho)
sigma = cov_matrix + np.outer(mu, mu)
_d_rs = Generate_d_rs(mu, cov_matrix, 1000)
d_rs = {f'{k}': d_r for k, d_r in enumerate(_d_rs)}
e_param = {'m': m,
           'n': n,
           'f': f,
           'h': h,
           'kappa': 1.0,
           'graph': graph,
           'mu': mu,
           'sigma': sigma,
           'cv': cv,
           'rho': rho,
           'd_rs': d_rs}

# saa_param
_saa_d_rs = Generate_d_rs(mu, cov_matrix, 30)
_saa_d_rs = {f'{k}': saa_d_r for k, saa_d_r in enumerate(_saa_d_rs)}
saa_param = {'d_rs': _saa_d_rs}

# co_param
co_param = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
            'bb_params': {'find_init_z': 'v1',
                          'select_branching_pos': 'v1'}}

# mv_param
mv_param = deepcopy(co_param)
