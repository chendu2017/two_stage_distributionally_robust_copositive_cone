from copy import deepcopy
import pandas as pd
from numerical_study.experiment import Experiment
import numpy as np
from numerical_study.ns_utils import Write_Output, Run_CO, Run_MV, Run_SAA

seed = 5
# driver
'''
cv, rho, kappa, OBSERVATION_SIZE, CI 
会改变输出文件的名字
'''
RHOs = [-1]
CVs = [-1]
KAPPAs = [-1]

OBSERVATION_DIST, OBSERVATION_SIZE = -1, 20
IN_SAMPLE_Drs_DIST, IN_SAMPLE_Drs_SIZE = OBSERVATION_DIST, 1000
OUT_SAMPLE_Drs_DIST, OUT_SAMPLE_Drs_SIZE = -1, 1000

BB_VERSION = 'v2'

BOOTSTRAP_CI = 30
REPLICATES = 10000

case_study_path = './case_study'
graph = pd.read_excel(case_study_path + '/input/graph.xlsx', index_col=0).values
m, n = graph.shape

# 单周租金成本 = 单价*平方*七天
f = (np.asarray([4.26, 6.36, 2.21, 5.39, 3.29, 3.82, 3.79]) * 200 * 7).tolist()
# SKU 343738 单价469，公司内部10%库存折损率+6%占用资金利率 = 75.04  平摊到12周 46.9/12 = 6.25元/周 + 10% noise
np.random.seed(seed)
h = np.random.uniform(low=6.25 * 0.9, high=6.25 * 1.1, size=m).tolist()

g = 'case_study'


def Construct_Task_Params():
    task_params = []

    for rho, cv, kappa in zip(RHOs, CVs, KAPPAs):
        e_param = {
            # e path
            'e_path': case_study_path,

            # graph setting
            'g': g,
            'm': m,
            'n': n,
            'graph': graph,
            'f': f,
            'h': h,
            'mu': [-1]*n,
            'rho': rho,
            'cv': cv,
            'kappa': kappa,

            # demand setting
            'seed': seed,
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


if __name__ == '__main__':
    task_params = Construct_Task_Params()
    task_param = task_params[0]
    e_param, co_param, mv_param, saa_param \
        = task_param['e_param'], task_param['co_param'], task_param['mv_param'], task_param['saa_param']
    dir_path = e_param['e_path']

    e = Experiment(e_param)
    demand_observations = pd.read_excel(case_study_path + './input/sku2_2017_sales.xlsx', index_col=0).values
    e.Set_Demand_Observations(demand_observations)
    e.d_rs_insample = e.demand_observations
    e.d_rs_outsample = pd.read_excel(case_study_path + './input/sku2_2018_sales.xlsx', index_col=0).values

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

