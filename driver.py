import os
from copy import deepcopy

import json

from numerical_study.experiment import Experiment
from numerical_study.ns_utils import Chunks, Run_CO, Write_Output, Run_MV, Run_SAA
import time
from numerical_study.SETTINGS import RHOs, CVs, KAPPAs, OBSERVATION_DIST, OBSERVATION_SIZE
from numerical_study.SETTINGS import IN_SAMPLE_Drs_DIST, IN_SAMPLE_Drs_SIZE, OUT_SAMPLE_Drs_DIST, OUT_SAMPLE_Drs_SIZE
from numerical_study.SETTINGS import BB_VERSION, BOOTSTRAP_CI, REPLICATES
from concurrent import futures


def Construct_Task_Params():
    task_params = []
    for m, n in [(6, 6)]:
        for g in range(30):
            with open(f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/'
                      f'{m}{n}/graph_{g}/graph_setting.txt') as file_gs:
                graph_setting = json.loads(file_gs.readline())
            for rho, cv, kappa in zip(RHOs, CVs, KAPPAs):
                seed = int(time.time())
                e_param = {
                    # global randomization seed
                    'seed': seed,

                    # e path
                    'e_path': f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/'
                                f'{m}{n}/graph_{g}',

                    # graph setting
                    'g': g,
                    'm': m,
                    'n': n,
                    'graph': graph_setting['graph'],
                    'f': graph_setting['f'],
                    'h': graph_setting['h'],
                    'mu': graph_setting['mu'],
                    'rho': rho,
                    'cv': cv,
                    'kappa': kappa,

                    # demand setting
                    'demand_observations_dist': OBSERVATION_DIST,
                    'demand_observations_sample_size': OBSERVATION_SIZE,
                    'in_sample_demand_dist': IN_SAMPLE_Drs_DIST,
                    'in_sample_demand_sample_size': IN_SAMPLE_Drs_SIZE,
                    'out_sample_demand_dist': OUT_SAMPLE_Drs_DIST,
                    'out_sample_demand_sample_size': OUT_SAMPLE_Drs_SIZE,

                    # bootstrap setting
                    'bootstrap_CI': BOOTSTRAP_CI,
                    'replicates': REPLICATES,
                }

                co_param = {
                        'bb_params': {'find_init_z': BB_VERSION,
                                      'select_branching_pos': BB_VERSION},
                        # bootstrap setting
                        'bootstrap_CI': e_param['bootstrap_CI'],
                        'replicates': e_param['replicates'],
                }
                mv_param = deepcopy(co_param)
                saa_param = {}

                task_param = {'e_param': e_param,
                              'co_param': co_param,
                              'mv_param': mv_param,
                              'saa_param': saa_param}
                task_params.append(task_param)
    return task_params


def Run_Single_Task(task_param):
    e_param, co_param, mv_param, saa_param \
        = task_param['e_param'], task_param['co_param'], task_param['mv_param'], task_param['saa_param']

    m, n, g = e_param['m'], e_param['n'], e_param['g']
    # output directory
    dir_path = e_param['e_path']
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    e = Experiment(e_param)

    # co
    co_output = Run_CO(e, co_param)
    Write_Output(dir_path, co_output)
    print(f'----{(m, n)}----graph{g}----CO----' + 'Done----')

    # mv
    mv_output = Run_MV(e, mv_param)
    Write_Output(dir_path, mv_output)
    print(f'----{(m, n)}----graph{g}----MV----' + 'Done----')

    # saa
    saa_output = Run_SAA(e, saa_param)
    Write_Output(dir_path, saa_output)
    print(f'----{(m, n)}----graph{g}----SAA----' + 'Done----')

    return f'----{(m, n)}----graph{g}----' + 'DoneDoneDoneDone----'


if __name__ == '__main__':
    task_params = Construct_Task_Params()
    # Run_Single_Task(task_params[0])

    try:
        for task_params in Chunks(task_params, 50):
            print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
            with futures.ProcessPoolExecutor(max_workers=4) as executor:
                tasks = [executor.submit(Run_Single_Task, task_param) for task_param in task_params]
                for task in futures.as_completed(tasks):
                    task_return = task.result()
                    print(task_return)
    except Exception as e:
        print(e)
