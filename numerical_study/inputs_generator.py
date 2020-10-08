import numpy as np
import json

from numerical_study.ns_utils import Generate_Graph, Modify_mu, Construct_Numerical_Input

if __name__ == '__main__':
    """
    在
    rhos = [-0.2, 0, 0.2]
    cvs = [0.1, 0.3, 0.5]
    in_sample_sizes = [30]
    基础上，扩充 setting时，需要修改 k_start
    
    扩充后的suffix_index, inputs 会首先出现在 new_inputs文件夹里
    确认无误后，再合并到主文件夹
    """
    suffix_index = {}
    k_start = 0
    rhos = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    cvs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    kappas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    base_rho, base_cv, base_kappa = 0.3, 0.3, 1.0

    triple = [[base_rho, base_cv, base_kappa]]
    for rho in list(set(rhos) - {base_rho}):
        triple.append([rho, base_cv, base_kappa])
    for cv in list(set(cvs) - {base_cv}):
        triple.append([base_rho, cv, base_kappa])
    for kappa in list(set(kappas) - {base_kappa}):
        triple.append([base_rho, base_cv, kappa])

    for m, n in [(6, 6)]:

        # 50 graphs
        for _g in range(50):
            f = np.random.randint(50, 100, m).tolist()
            h = np.round(np.random.uniform(0.1, 0.2, m), 3).tolist()
            _num_arcs = int(m * 2.4)  # 3 <-- 2018 ShuJia POMS; PS. int has the same function as floor in this case
            # randomly generate a graph having num_arcs links, and for each location, at least one link is spawned.
            graph = Generate_Graph(m, n, _num_arcs)
            mu = [500] * n
            mu = Modify_mu(mu, epsilon=0.3)
            # save graph setting
            _graph_setting = {'graph': graph,
                              'f': f,
                              'h': h,
                              'mu': mu}

            dir_path = f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs/{m}{n}/graph{_g}'
            with open(dir_path + '/graph_setting.txt', 'w') as file_gs:
                file_gs.write(json.dumps(_graph_setting))

            _k = k_start
            print('m, n:', m, n, 'graph:', _g)
            for _k, (rho, cv, kappa) in enumerate(triple):
                file_name = dir_path + f'/input/input{_k}.txt'

                # calculate sigma and generate d_rs according to (rho, cv). Then save the input param

                numerical_input = Construct_Numerical_Input(m, n, f, h, graph, mu, rho, cv, kappa)
                with open(file_name, 'w') as f_write:
                    f_write.write(json.dumps(numerical_input))

                # build k-suffix_index
                suffix_index.update({f'{_k}': {'rho': rho,
                                               'cv': cv,
                                               'kappa': kappa}})

                _k += 1

    # record suffix info
    suffix_index_path = f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs/suffix_index.txt'
    with open(suffix_index_path, 'r') as f_read:
        indices = json.loads(f_read.readline())
    with open(suffix_index_path, 'w') as f_write:
        indices.update(suffix_index)
        f_write.write(json.dumps(indices))
