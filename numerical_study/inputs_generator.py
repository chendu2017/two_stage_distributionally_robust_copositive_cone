from copy import deepcopy
import numpy as np
import json

from numerical_study.ns_utils import Generate_Graph, Calculate_Cov_Matrix, Generate_d_rs, Construct_Numerical_Input, Modify_mus

if __name__ == '__main__':

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
            with open(f'D:/【论文】NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/graph_setting.txt', 'w') as file_gs:
                file_gs.write(json.dumps(_graph_setting))

            mu = 500
            _k = 0
            print('m, n:', m, n, 'graph:', _g)
            for rho in [-0.2, 0, 0.2]:
                for cv in [0.1, 0.3, 0.5]:
                    for in_sample_size in [5, 10, 20, 30]:

                        # equal_mean
                        mus = Modify_mus(n, mu, epsilon=0.0)
                        cov_matrix = Calculate_Cov_Matrix(mus, cv, rho)
                        d_rs = Generate_d_rs(mus, cov_matrix, in_sample_size)
                        outsample_d_rs = Generate_d_rs(mus, cov_matrix, 1000)
                        numerical_input = Construct_Numerical_Input(m, n, f, h, graph, d_rs, outsample_d_rs)
                        with open(f'D:/【论文】NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/equal_mean/input/input{_k}.txt', 'w') \
                                as f_equal_mean:
                            f_equal_mean.write(json.dumps(numerical_input))
                        del mus, cov_matrix, d_rs, outsample_d_rs, numerical_input

                        # non_equal_mean
                        epsilon = 0.1
                        mus = Modify_mus(n, mu, epsilon)
                        cov_matrix = Calculate_Cov_Matrix(mus, cv, rho)
                        d_rs = Generate_d_rs(mus, cov_matrix, in_sample_size)
                        outsample_d_rs = Generate_d_rs(mus, cov_matrix, 1000)
                        numerical_input = Construct_Numerical_Input(m, n, f, h, graph, d_rs, outsample_d_rs)
                        with open(f'D:/【论文】NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean/input/input{_k}.txt', 'w') \
                                as f_non_equal_mean:
                            f_non_equal_mean.write(json.dumps(numerical_input))
                        del mus, cov_matrix, d_rs, outsample_d_rs, numerical_input

                        # non_equal_mean_mixture_Gaussian
                        epsilons = [0.0, 0.05, 0.10, 0.15, 0.20]
                        num_component = len(epsilons)
                        d_rs, outsample_d_rs = [], []
                        for e in epsilons:
                            mus = Modify_mus(n, mu, e)
                            cov_matrix = Calculate_Cov_Matrix(mus, cv, rho)
                            d_rs += Generate_d_rs(mus, cov_matrix, int(in_sample_size/num_component))
                            outsample_d_rs += Generate_d_rs(mus, cov_matrix, int(1000/num_component))
                        numerical_input = Construct_Numerical_Input(m, n, f, h, graph, d_rs, outsample_d_rs)
                        with open(
                                f'D:/【论文】NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean_mixture_gaussian/input/input{_k}.txt',
                                'w') \
                                as f_non_equal_mean_mixture_gaussian:
                            f_non_equal_mean_mixture_gaussian.write(json.dumps(numerical_input))
                        del mus, cov_matrix, d_rs, outsample_d_rs, numerical_input

                        _k += 1

