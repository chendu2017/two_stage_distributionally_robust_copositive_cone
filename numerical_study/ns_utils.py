import time
from typing import List

from numpy.random import choice
import numpy as np
from copy import deepcopy
import json
from scipy.stats import lognorm, truncnorm, multivariate_normal, uniform

TOL = 1e-7


def Generate_Graph(m, n, num_arcs, seed=int(time.time())) -> List[List[int]]:
    """
    automatically generate a bipartite graph with given number of arcs such that
    each location has at least on link and each POD has at least one link as well.
    :param seed: randomization seed
    :param m: # of locations
    :param n: # of POD
    :param num_arcs: # of arcs
    :return:
    """
    tmp_graph = np.asarray([1] * num_arcs + [0] * (m * n - num_arcs))
    while True:
        seed += 1
        np.random.seed(seed)
        np.random.shuffle(tmp_graph)
        graph = np.reshape(tmp_graph, [m, n])
        if all(graph.sum(axis=0)) and all(graph.sum(axis=1)):
            graph = graph.tolist()
            return graph


def Generate_d_rs(mus, cv, rho, sample_size,
                  dist=None, seed=int(time.time())) -> np.ndarray:
    """
    dist:  -1: None
            1: multivariate-normal,
            2: two_point,
            3: truncated independent normal
            4: truncated uniform
            5: truncated lognorm
    """
    if dist == -1:
        d_rs = -np.ones(shape=(2, 2))
        return d_rs

    if dist is None:
        dist = [1]

    def Generate_d_rs_with_RV(rv, size):
        d_rs = []
        current_size = 0
        while current_size < size:
            rv_realized = rv.rvs()
            if all(rv_realized >= 0):
                d_rs.append(rv_realized)
                current_size += 1
        return np.asarray(d_rs)

    n = len(mus)
    mus, stds = np.asarray(mus), np.asarray([mus[i] * cv for i in range(n)])
    d_rs = None

    # if sample_size is a list, equally draw demand realizations from these dists
    if not isinstance(dist, int):
        dists = deepcopy(dist)
        component_size = int(sample_size / len(dists))
        d_rs = np.concatenate([Generate_d_rs(mus, cv, rho, component_size, dist, seed) for dist in dists])

    if dist == 1:
        # 1: multivariate gaussian
        cov_matrix = Calculate_True_Cov_Matrix(mus, cv, rho)
        rv = multivariate_normal(mus, cov_matrix, seed=seed)
        d_rs = Generate_d_rs_with_RV(rv, sample_size)

    if dist == 2:
        # 2: two_points
        np.random.seed(seed)
        choices = choice([-1, 1], size=sample_size, replace=True)
        d_rs = np.asarray([mus + stds * _ for _ in choices])

    if dist == 3:
        # 2: truncated independent normal
        a, b = (0 - mus) / stds, (float('inf') - mus) / stds
        rv = truncnorm(a=a, b=b, loc=mus, scale=stds)
        rv.random_state = np.random.RandomState(seed=seed)
        d_rs = Generate_d_rs_with_RV(rv, sample_size)

    if dist == 4:
        # 4: uniform
        rv = uniform(loc=mus - np.sqrt(3) * stds, scale=2 * np.sqrt(3) * stds)  # U[loc, loc+scale]
        rv.random_state = np.random.RandomState(seed=seed)
        d_rs = Generate_d_rs_with_RV(rv, sample_size)

    if dist == 5:
        # 5: log normal
        lognorm_variance = np.log((stds / mus) ** 2 + 1)
        lognorm_mus = np.log(mus) - 0.5 * lognorm_variance
        lognorm_stds = np.sqrt(lognorm_variance)
        rv = lognorm(s=lognorm_stds, scale=np.exp(lognorm_mus))
        rv.random_state = np.random.RandomState(seed)
        d_rs = Generate_d_rs_with_RV(rv, sample_size)

    # clip
    d_rs[d_rs <= 0] = 0
    return d_rs


def Calculate_True_Cov_Matrix(mus, cv, rho):
    n = len(mus)
    cov_matrix = np.zeros([n, n])
    stds = [mu * cv for mu in mus]
    for i in range(n):
        for j in range(n):
            cov_matrix[i][j] = stds[i] * stds[j] * rho
            if i == j:
                cov_matrix[i][j] = stds[i] ** 2
    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
    if min_eig < 0:
        cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)
    return cov_matrix


def Calculate_True_Sigma_Matrix(mus, cv, rho):
    cov_m = Calculate_True_Cov_Matrix(mus, cv, rho)
    sigma = cov_m + np.outer(mus, mus)
    return sigma


def Calculate_Sampled_Cov_Matrix(samples):
    if isinstance(samples, dict):
        samples = np.asarray([sample for k, sample in samples.items()])
    cov_sampled = np.cov(samples.T)
    return cov_sampled


def Calculate_Sampled_Sigma_Matrix(samples):
    if isinstance(samples, dict):
        samples = np.asarray([sample for k, sample in samples.items()])
    sigma_sampled = np.matmul(samples.T, samples) / samples.shape[0]
    return sigma_sampled


def Modify_mu(mu, epsilon):
    ret = [_ + _ * np.random.uniform(-epsilon, epsilon) for _ in mu]
    return ret


def Write_Output(dir_path, output):
    model = output['model']
    cv, rho, kappa = output['e_param']['cv'], output['e_param']['rho'], output['e_param']['kappa']
    observations = output['e_param']['demand_observations_sample_size']
    CI = output['e_param']['bootstrap_CI']
    file_path = dir_path + f'/{model}_({cv},{rho},{kappa})_N={observations}_CI={CI}.txt'
    with open(file_path, 'w') as f:
        f.write(json.dumps(output))


def Chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def isAllInteger(numbers):
    allIntegerFlag = all(map(isZeroOneInteger, numbers))
    return allIntegerFlag


def isZeroOneInteger(x):
    return abs(x - 1) <= TOL or abs(x) <= TOL


if __name__ == '__main__':
    mus = [20, 20, 20, 20]
    cv, rho = 0.1, 0.1
    simga = Calculate_True_Sigma_Matrix(mus, cv, rho)
    cov = Calculate_True_Cov_Matrix(mus, cv, rho)
    sample_size = 1000

    cov_matrix = Calculate_True_Cov_Matrix(mus, cv, rho)
    d_rs = Generate_d_rs(mus=mus, cv=cv, rho=rho, sample_size=sample_size, dist=[1])
    print(cov_matrix)
    print(np.cov(np.asarray(d_rs).T))
    print(np.asarray(d_rs).mean(axis=0))

    # Remove_Input('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')
    # Remove_Output('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')