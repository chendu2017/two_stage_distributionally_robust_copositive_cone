import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from copy import deepcopy

import numpy as np
import json
from numerical_study.experiment import Experiment
from numerical_study.ns_utils import Chunks
from concurrent import futures

bootstrap_CI = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
REPLICATES = 10000


def Calculate_Graph():
    graph = [[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, ],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, ],
             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, ],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ],
             [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, ],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, ],
             [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, ],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, ]]
    graph = np.asarray(graph)
    return graph


def Write_Output(dir_path, output):
    model = output['model']
    file_path = dir_path
    K = output['e_param']['demand_observations_sample_size']
    rho = output['e_param']['rho']

    if model in ['mv', 'co']:
        CI = output['algo_param']['bootstrap_CI']
        file_path = dir_path + f'output/{model}_CI={CI}_K={K}_rho={rho}.txt'

    if model == 'saa':
        file_path = dir_path + f'output/{model}_K={K}.txt'

    if model == 'wass':
        r = output['algo_param']['wasserstein_ball_radius']
        p = output['algo_param']['wasserstein_p']
        K = output['algo_param']['max_num_extreme_points']
        file_path = dir_path + f'output/{model}_r={r}_p={p}_K={K}.txt'

    with open(file_path, 'w') as f:
        f.write(json.dumps(output))


def Construct_Task_Params():
    task_params = []

    graph = Calculate_Graph()
    m, n = graph.shape
    for rho, K in zip([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [0.7] * 4, [20] * 9 + [10, 30, 40, 50]):
            e_param = {
                # global randomization seed
                'seed': 742511,

                # e path
                'e_path': './',

                # graph setting
                'g': 'case_study_yaan',
                'm': m,
                'n': n,
                'graph': graph.tolist(),
                'f': [51.26, 47.23, 50.47, 48.88, 42.11, 49.27, 44.48, 44.82, 42.96, 35.11, 46.76, 41.80],
                'h': [0.0289, 0.0279, 0.0278, 0.0282, 0.0286, 0.0263, 0.0280, 0.0292, 0.0294, 0.0295, 0.0283, 0.0264],
                'mu': [980, 1231, 3992, 4068, 395, 762, 530, 394, 448, 400, 602, 326],
                'rho': rho,
                'cv': 0.7,
                'kappa': 1,

                # demand setting
                'demand_observations_dist': 1,
                'demand_observations_sample_size': K,
                'in_sample_demand_dist': 1,
                'in_sample_demand_sample_size': 1000,
                'out_sample_demand_dist': [1, 2, 3, 4, 5],
                'out_sample_demand_sample_size': 1000,
            }

            # algo params
            co_params = []
            mv_params = []
            for ci in bootstrap_CI:
                co_param = {
                    'model': 'co',
                    'bb_params': {'find_init_z': 'v3',
                                  'select_branching_pos': 'v2'},
                    # bootstrap setting
                    'bootstrap_CI': ci,
                    'replicates': REPLICATES,
                }
                co_params.append(co_param)
                mv_param = deepcopy(co_param)
                mv_param.update({'model': 'mv'})
                mv_params.append(mv_param)
            saa_param = {'model': 'saa'}

            task_param = {'e_param': e_param,
                          'algo_params': [saa_param] + co_params + mv_params
                          }
            task_params.append(task_param)
    return task_params


if __name__ == '__main__':
    task_params = Construct_Task_Params()
    es = []
    algo_params = []
    for task_param in task_params:
        e_param = task_param['e_param']

        m, n, g = e_param['m'], e_param['n'], e_param['g']
        # output directory
        dir_path = e_param['e_path']
        if not os.path.exists(dir_path + 'output/'):
            os.makedirs(dir_path + 'output/')

        e = Experiment(e_param)

        es += [e] * len(task_param['algo_params'])
        algo_params += task_param['algo_params']

    # for each algo_param, run model
    try:
        with futures.ProcessPoolExecutor(max_workers=4) as executor:
            tasks = [executor.submit(es[i].Run, algo_param) for i, algo_param in enumerate(algo_params)]
            for k, task in enumerate(futures.as_completed(tasks)):
                task_return = task.result()
                Write_Output('./', task_return)
                print(f'----{task_return["model"]}----algo_{k}----' + 'Done----')
    except Exception as error:
        print(error)
