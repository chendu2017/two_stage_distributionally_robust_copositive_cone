from copy import deepcopy
import numpy as np
import json

from numerical_study.ns_utils import Generate_Graph, Generate_Gaussian_Input

if __name__ == '__main__':
    suffix_index = {}
    k_start = 0  # 从k_start开始， 总计数会增加 rhos * cvs * sizes 个数
    rhos = [-0.2, 0, 0.2]
    cvs = [0.1, 0.3, 0.5]
    in_sample_sizes = [30]

    for m, n in [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:

        # 20 graphs
        for _g in range(20):
            f = np.random.randint(50, 100, m).tolist()
            h = np.round(np.random.uniform(0.1, 0.2, m), 3).tolist()
            _num_arcs = int(m * 3)  # 3 <-- 2018 ShuJia POMS; PS. int has the same function as floor in this case
            # randomly generate a graph having num_arcs links, and for each location, at least one link is spawned.
            graph = Generate_Graph(m, n, _num_arcs)
            # save graph setting
            _graph_setting = {'graph': graph,
                              'f': f,
                              'h': h}

            dir_path = f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs/{m}{n}/graph{_g}'
            with open(dir_path + '/graph_setting.txt', 'w') as file_gs:
                file_gs.write(json.dumps(_graph_setting))

            mu = 500
            _k = k_start
            print('m, n:', m, n, 'graph:', _g)
            for rho in rhos:
                for cv in cvs:
                    for in_sample_size in in_sample_sizes:
                        for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                            file_name = dir_path + f'/{mode}/input/input{_k}.txt'

                            if mode == 'equal_mean':
                                Generate_Gaussian_Input(file_name, m, n, f, h, graph, mu, rho, cv, in_sample_size,
                                                        epsilons=[0.0])

                            if mode == 'non_equal_mean':
                                Generate_Gaussian_Input(file_name, m, n, f, h, graph, mu, rho, cv, in_sample_size,
                                                        epsilons=[0.1])

                            if mode == 'non_equal_mean_gaussian':
                                Generate_Gaussian_Input(file_name, m, n, f, h, graph, mu, rho, cv, in_sample_size,
                                                        epsilons=[0.0, 0.05, 0.10, 0.15, 0.20])

                        # build k-suffix_index
                        suffix_index.update({f'{_k}': {'rho': rho,
                                                       'cv': cv,
                                                       'in_sample:size': in_sample_size}})

                        _k += 1

    # record suffix info
    suffix_index_path = f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs/suffix_index.txt'
    with open(suffix_index_path, 'r') as f:
        indices = json.loads(f.readline())
    with open(suffix_index_path, 'w') as f:
        indices.update(suffix_index)
        f.write(json.dumps(indices))
