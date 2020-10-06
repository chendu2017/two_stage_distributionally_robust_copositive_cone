from typing import List, Dict, Any
import os
import numpy as np
from copy import deepcopy
import json


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


def Generate_Gaussian_Input(file_name, m, n, f, h, graph, mu, rho, cv, in_sample_size, epsilons=[0.0]):
    num_component = len(epsilons)
    d_rs, outsample_d_rs = [], []
    mus = []
    for e in epsilons:
        mus = Modify_mus(n, mu, e)
        cov_matrix = Calculate_Cov_Matrix(mus, cv, rho)
        d_rs += Generate_d_rs(mus, cov_matrix, in_sample_size // num_component)
        outsample_d_rs += Generate_d_rs(mus, cov_matrix, 1000 // num_component)
    numerical_input = Construct_Numerical_Input(m, n, f, h, graph, mus,
                                                rho, cv, in_sample_size, d_rs, outsample_d_rs)

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


def Construct_Algo_Params():
    # co
    co_params = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                 'bb_params': {'find_init_z': 'v1',
                               'select_branching_pos': 'v1'}}
    co_speedup_params = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                         'bb_params': {'find_init_z': 'v2',
                                       'select_branching_pos': 'v2'}}
    # mv
    mv_params = deepcopy(co_speedup_params)
    # saa
    saa_params = {}

    return co_params, co_speedup_params, mv_params, saa_params


def Construct_Numerical_Input(m, n, f, h, graph, mus, rho, cv, in_sample_size, d_rs, outsample_d_rs) -> Dict[str, Any]:
    d_rs = {k: d_r for k, d_r in enumerate(d_rs)}
    outsample_d_rs = {k: d_r for k, d_r in enumerate(outsample_d_rs)}
    e_params = e_params = {'m': m,
                           'n': n,
                           'f': f,
                           'h': h,
                           'graph': graph,
                           'mus': mus,
                           'rho': rho,
                           'cv': cv,
                           'in_sample_size': in_sample_size,
                           'd_rs': d_rs,
                           'outsample_d_rs': outsample_d_rs}

    co_params, co_speedup_params, mv_params, saa_params = Construct_Algo_Params()
    ret = {'e_params': e_params,
           'co_params': co_params,
           'co_speedup_params': co_speedup_params,
           'mv_params': mv_params,
           'saa_params': saa_params}
    return ret


def Modify_mus(n, mu, epsilon):
    ret = [mu + np.random.uniform(-epsilon, epsilon) for _ in range(n)]
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
    c, cs, mv, saa = Construct_Algo_Params()
    print(c, cs, mv, saa)

    g = Generate_Graph(5, 5, 10)
    print(g)

    cov_matrix = Calculate_Cov_Matrix([20, 30, 40, 50, 60], 0.1, 0.1)
    d_rs = Generate_d_rs([20, 30, 40, 50, 60], cov_matrix, 100000)
    print(np.cov(np.asarray(d_rs).T))
    print(np.asarray(d_rs).mean(axis=0))
    
    Remove_Input('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')
    Remove_Output('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs')
