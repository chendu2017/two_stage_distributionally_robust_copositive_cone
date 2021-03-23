
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from copy import deepcopy

import numpy as np
import json
import pandas as pd
from numerical_study.experiment import Experiment
from numerical_study.ns_utils import Chunks
import time
from concurrent import futures

rider_speed = 18  # km/hour
Aplus_time_limit = 1
allowed_rider_travel_time = 1

bootstrap_CI = 0
REPLICATES = 10000


def Calculate_Graph():
    distance = pd.read_excel('./input/network_distance.xlsx', index_col=0)
    shangquan_1hr_arrival = ['南京东路', '静安寺', '徐家汇', '陆家嘴', '五角场']
    shangquan_1hr_arrival_cols = []
    for k, col in enumerate(distance.columns):
        if col in shangquan_1hr_arrival:
            shangquan_1hr_arrival_cols.append(k)

    travel_time = distance.values / rider_speed
    graph = np.where(travel_time <= 2, 1, 0)
    for k in shangquan_1hr_arrival_cols:
        graph[:, k] = np.where(travel_time[:, k] <= Aplus_time_limit, 1, 0)
    return graph


def Write_Output(dir_path, output):
    model = output['model']
    file_path = dir_path

    if model in ['mv', 'co']:
        CI = output['algo_param']['bootstrap_CI']
        REP = output['algo_param']['replicates']
        file_path = dir_path + f'output/{model}_CI={CI}_rep={REP}.txt'

    if model == 'saa':
        file_path = dir_path + f'output/{model}.txt'

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
    e_param = {
        # global randomization seed
        'seed': int(time.time()),

        # e path
        'e_path': './',

        # graph setting
        'g': 'case_study',
        'm': m,
        'n': n,
        'graph': graph.tolist(),
        'f': (np.asarray([0.39738806, 0.593283582, 0.206156716, 0.502798507, 0.306902985, 0.356343284, 0.353544776])*40*0.173).tolist(),
        'h': [0.070891085, 0.065846098, 0.066436721, 0.064461182, 0.073795667, 0.074853593, 0.065543422],
        'mu': [1] * n,
        'rho': 0,
        'cv': 0.1,
        'kappa': 1,

        # demand setting
        'demand_observations_dist': -1,
        'demand_observations_sample_size': -1,
        'in_sample_demand_dist': -1,
        'in_sample_demand_sample_size': -1,
        'out_sample_demand_dist': -1,
        'out_sample_demand_sample_size': -1,
    }

    # algo params
    co_param = {
        'model': 'co',
        'bb_params': {'find_init_z': 'v3',
                      'select_branching_pos': 'v2'},
        # bootstrap setting
        'bootstrap_CI': bootstrap_CI,
        'replicates': REPLICATES,
    }

    mv_param = deepcopy(co_param)
    mv_param.update({'model': 'mv'})

    saa_param = {'model': 'saa'}

    wass_param = {'model': 'wass',
                  'wasserstein_ball_radius': 10,
                  'wasserstein_p': 1,
                  'max_num_extreme_points': 10000}

    task_param = {'e_param': e_param,
                  'algo_params': [mv_param]
                  }
    task_params.append(task_param)
    return task_params


if __name__ == '__main__':
    task_param = Construct_Task_Params()[0]

    e_param = task_param['e_param']

    m, n, g = e_param['m'], e_param['n'], e_param['g']
    # output directory
    dir_path = e_param['e_path']
    if not os.path.exists(dir_path + 'output/'):
        os.makedirs(dir_path + 'output/')

    e = Experiment(e_param)

    # put real data
    demand_obsers = pd.read_excel(dir_path + 'input/sku2_343738/sku2_2017_sales.xlsx', index_col=0).values
    demand_obsers_outsample = pd.read_excel(dir_path + 'input/sku2_343738/sku2_2018_sales.xlsx', index_col=0).values
    e.Set_Demand_Observations(demand_obsers)
    e.d_rs_insample = demand_obsers
    e.d_rs_outsample = demand_obsers_outsample

    # for each algo_param, run model
    try:
        for algo_params in Chunks(task_param['algo_params'], 50):
            print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
            with futures.ProcessPoolExecutor(max_workers=2) as executor:
                tasks = [executor.submit(e.Run, algo_param) for algo_param in algo_params]
                for task in futures.as_completed(tasks):
                    task_return = task.result()
                    Write_Output(dir_path, task_return)
                    print(f'----{(m, n)}----graph{g}----{task_return["model"]}----' + 'Done----')
    except Exception as e:
        print(e)
