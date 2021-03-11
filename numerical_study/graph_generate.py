from copy import deepcopy

import numpy as np
import json
import os
from numerical_study.ns_utils import Generate_Graph, Modify_mu
from numerical_study.SETTINGS import F_LB, F_UB, H_LB, H_UB, MU, MU_EPSILON


def Generate_Graph_Setting():
    for m, n in [(6, 6)]:
        # 50 graphs
        for _g in range(30):
            f = np.random.randint(F_LB, F_UB, m).tolist()
            h = np.round(np.random.uniform(H_LB, H_UB, m), 3).tolist()
            _num_arcs = int(
                (m + n) * 1.2)  # 3 <-- 2018 ShuJia POMS; PS. int has the same function as floor in this case
            # randomly generate a graph having num_arcs links, and for each location, at least one link is spawned.
            graph = Generate_Graph(m, n, _num_arcs)
            mu = [MU] * n
            mu = Modify_mu(mu, epsilon=MU_EPSILON)
            # save graph setting
            _graph_setting = {'graph': graph,
                              'f': f,
                              'h': h,
                              'mu': mu}

            dir_path = f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph_{_g}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # graph setting file
            with open(dir_path + '/graph_setting.txt', 'w') as file_gs:
                file_gs.write(json.dumps(_graph_setting))


if __name__ == '__main__':
    Generate_Graph_Setting()
