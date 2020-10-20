from typing import List, Dict, Any
import os
import numpy as np
from copy import deepcopy
import json
from deprecated import deprecated
from scipy.stats import lognorm


def Generate_Graph(m, n, num_arcs) -> List[List[int]]:
    """
    automatically generate a bipartite graph with given number of arcs such that
    each location has at least on link and each POD has at least one link as well.
    :param m: # of locations
    :param n: # of POD
    :param num_arcs: # of arcs
    :return:
    """
    tmp_graph = np.asarray([1] * num_arcs + [0] * (m * n - num_arcs))
    while True:
        np.random.shuffle(tmp_graph)
        graph = np.reshape(tmp_graph, [m, n])
        if all(graph.sum(axis=0)) and all(graph.sum(axis=1)):
            graph = graph.tolist()
            return graph


def Generate_d_rs(mus, cov_matrix, in_sample_size) -> List[List[float]]:
    d_rs = np.random.multivariate_normal(mus, cov_matrix, size=in_sample_size)
    # clip
    d_rs[d_rs <= 0] = 0
    d_rs = d_rs.tolist()
    return d_rs


def Generate_d_rs_out_sample(mus, cov_matrix, cv, out_sample_size) -> List[List[float]]:
    n = len(mus)
    mus, cov_matrix = np.asarray(mus), np.asarray(cov_matrix)
    stds = np.sqrt([cov_matrix[i, i] for i in range(n)])

    component = 4
    component_size = int(out_sample_size/component)
    # 1: two_points
    d_rs_1 = np.asarray([mus-stds + np.random.randint(0, 2)*stds*2 for _ in range(component_size)])
    # 2: independent normal
    d_rs_2 = [np.random.normal(mus, stds) for _ in range(component_size)]
    # 3: uniform
    d_rs_3 = np.asarray([np.random.uniform(low=mus-np.sqrt(3)*stds, high=mus+np.sqrt(3)*stds) for _ in range(component_size)])
    # 4: log normal
    d_rs_4 = np.asarray([lognorm.rvs(s=0.31*(cv/0.33), scale=mu, size=component_size) for mu in mus]).T
    d_rs = np.concatenate([d_rs_1, d_rs_2, d_rs_3, d_rs_4])
    # clip
    d_rs[d_rs <= 0] = 0
    d_rs = d_rs.tolist()
    return d_rs

@deprecated(reason='Construct_Numerical_Input has changes; '
                   'possibly useful for mixture gaussian (extension) after revision')
def Generate_Gaussian_Input(file_name, m, n, f, h, graph, mu, rho, cv, kappa, epsilons=[0.0]):
    num_component = len(epsilons)
    d_rs = []
    mus = []
    cov_matrixs = []
    for e in epsilons:
        mu = Modify_mu(mu, e)
        mus.append(mu)
        cov_matrix = Calculate_Cov_Matrix(mu, cv, rho)
        cov_matrixs.append(cov_matrix)
        d_rs += Generate_d_rs(mu, cov_matrix, 1000 // num_component)
    avg_mu = np.asarray(mus).mean(axis=0).tolist()
    avg_sigma = np.asarray(cov_matrixs).sum(axis=0) / num_component ** 2 + np.outer(avg_mu, avg_mu)  # second-moment

    numerical_input = Construct_Numerical_Input(m, n, f, h, graph, avg_mu, avg_sigma, rho, cv, kappa, d_rs)

    with open(file_name, 'w') as f:
        f.write(json.dumps(numerical_input))


def Calculate_Cov_Matrix(mus, cv, rho):
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


def Construct_Algo_Params(mu, sigma):
    # co
    co_param = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                'bb_params': {'find_init_z': 'v1',
                              'select_branching_pos': 'v1'}}
    co_speedup_param = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                        'bb_params': {'find_init_z': 'v2',
                                      'select_branching_pos': 'v2'}}
    # mv
    mv_param = deepcopy(co_speedup_param)
    # saa
    cov_m = np.asarray(sigma) - np.outer(mu, mu)
    saa_d_rs = Generate_d_rs(mu, cov_m, 30)
    saa_d_rs = {f'{k}': d_r for k, d_r in enumerate(saa_d_rs)}
    saa_param = {'d_rs': saa_d_rs}
    # det
    det_param = {'mu': mu}

    return co_param, co_speedup_param, mv_param, saa_param, det_param


def Construct_Numerical_Input(m, n, f, h, graph, mu, rho, cv, kappa):
    cov_matrix = Calculate_Cov_Matrix(mu, cv, rho)
    d_rs = Generate_d_rs(mu, cov_matrix, 1000)
    d_rs_outsample = Generate_d_rs_out_sample(mu, cov_matrix, cv, 1000)
    d_rs = {k: d_r for k, d_r in enumerate(d_rs)}
    d_rs_outsample = {k: d_r for k, d_r in enumerate(d_rs_outsample)}
    sigma = (cov_matrix + np.outer(mu, mu)).tolist()

    e_param = {'m': m,
               'n': n,
               'f': f,
               'h': h,
               'graph': graph,
               'mu': mu,
               'sigma': sigma,
               'rho': rho,
               'cv': cv,
               'kappa': kappa,
               'd_rs': d_rs,
               'd_rs_outsample': d_rs_outsample}

    co_param, co_speedup_param, mv_param, saa_param, det_param = Construct_Algo_Params(mu, sigma)
    ret = {'e_param': e_param,
           'co_param': co_param,
           'co_speedup_param': co_speedup_param,
           'mv_param': mv_param,
           'saa_param': saa_param,
           'det_param': det_param}

    return ret


def Modify_mu(mu, epsilon):
    ret = [_ + _ * np.random.uniform(-epsilon, epsilon) for _ in mu]
    return ret


def Write_Output(dir_path, output, k):
    model = output['model']
    file_path = dir_path + f'/output{k}_{model}.txt'
    with open(file_path, 'w') as f:
        f.write(json.dumps(output))


def Remove_Input(path):
    try:
        for m, n in [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:
            for g in range(20):
                for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                    input_path = path + f'/{m}{n}/graph{g}/{mode}/input'
                    file_lists = os.listdir(input_path)
                    for file in file_lists:
                        os.remove(input_path + f'/{file}')
    except FileNotFoundError:
        pass


def Remove_Output(path):
    try:
        for m, n in [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:
            for g in range(20):
                for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                    output_path = path + f'/{m}{n}/graph{g}/{mode}/output'
                    file_lists = os.listdir(output_path)
                    for file in file_lists:
                        os.remove(output_path + f'/{file}')
    except FileNotFoundError:
        pass


def Chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    mu = [20, 20, 20, 20]
    sigma = [[416., 403.2, 403.2, 403.2],
             [403.2, 416., 403.2, 403.2],
             [403.2, 403.2, 416., 403.2],
             [403.2, 403.2, 403.2, 416.]]
    c, cs, mv, saa = Construct_Algo_Params(mu, sigma)
    print(c, cs, mv, saa)

    g = Generate_Graph(5, 5, 10)
    print(g)

    cov_matrix = Calculate_Cov_Matrix([20, 30, 40, 50, 60], 0.1, 0.1)
    d_rs = Generate_d_rs([20, 30, 40, 50, 60], cov_matrix, 100000)
    d_rs_outsample = Generate_d_rs_out_sample([20, 30, 40, 50, 60], cov_matrix, 1000)
    print(cov_matrix)
    print(np.cov(np.asarray(d_rs).T))
    print(np.asarray(d_rs).mean(axis=0))

    # Remove_Input('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')
    # Remove_Output('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')
