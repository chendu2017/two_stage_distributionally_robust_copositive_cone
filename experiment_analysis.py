import json
from typing import Dict, List, Tuple
import re
import pandas as pd
from concurrent import futures
import os
import numpy as np

re_X = re.compile('\(([0-9]*), ([0-9]*)\)')
re_find_k = re.compile('output([0-9]+)_.*\.txt')
TOL = 1e-4


def Analyze_Simulation(m, n, I, simulations) -> List[Dict[str, float]]:
    """
    simulations: Dict[str, Dict[str, Any]]
    """
    rets = []
    for cnt, simulation in simulations.items():
        d_r = simulation['d_r']
        X = simulation['X']  # Dict[str, float] <- {'(1, 0)': 100.0, '(1, 1)': 200.0}

        sent, received = Parse_X(m, n, X)

        # metrics:
        # 1. supply: utilization_rate,
        # 2. demand: fulfill_rate, event_fulfill_rate, unmet_amount,
        utilization_rate = sum(sent) / sum(I)
        fulfill_rate = sum(received) / sum(d_r) if sum(d_r) != 0 else 1  # sum(d_r) possibly is zero
        event_fulfill_rate = sum([abs(r - d) <= TOL for r, d in zip(received, d_r)]) / n
        unmet_demand = sum(d_r) - sum(received)

        # construct return
        ret = {'utilization_rate': utilization_rate,
               'fulfill_rate': fulfill_rate,
               'event_fulfill_rate': event_fulfill_rate,
               'unmet_demand': unmet_demand,
               'cnt': cnt}

        rets.append(ret)
    return rets


def Parse_X(m, n, X: Dict[str, float]) -> Tuple[List[float], List[float]]:
    sent = [0] * m
    received = [0] * n
    for road, amount in X.items():
        re_X_ret = re_X.match(road)
        i, j = int(re_X_ret[1]), int(re_X_ret[2])
        sent[i] += amount
        received[j] += amount
    return sent, received


def Parse_Output(param) -> pd.DataFrame:
    path = param['path']
    m, n = param['m'], param['n']
    rho, cv, kappa = param['rho'], param['cv'], param['kappa']
    g = param['g']
    with open(path, 'r') as f:
        output = json.loads(f.readline())
    I = output['sol']['I']
    m = len(I)
    n = len(output['simulation']['0']['d_r'])
    simulation = output['simulation']
    simulation_outsample = output['simulation_outsample']
    sim_results = Analyze_Simulation(m, n, I, simulation)
    sim_results_outsample = Analyze_Simulation(m, n, I, simulation_outsample)

    # in-sample
    ret_df = pd.DataFrame(sim_results)
    ret_df['model'] = output['model']
    ret_df['obj'] = output['sol']['obj']
    ret_df['f'] = output['sol']['f']
    ret_df['h'] = output['sol']['h']
    # ret_df['I'] = json.dumps(output['sol']['I'])
    # ret_df['Z'] = json.dumps(output['sol']['Z'])
    ret_df['sum_I'] = sum(output['sol']['I'])
    ret_df['sum_Z'] = sum(output['sol']['Z'])
    ret_df['rho'] = rho
    ret_df['cv'] = cv
    ret_df['kappa'] = kappa
    ret_df['cpu_time'] = output['cpu_time']
    ret_df['graph'] = g

    # out-sample
    ret_df_outsample = pd.DataFrame(sim_results_outsample)
    ret_df = pd.merge(ret_df, ret_df_outsample[['utilization_rate',
                                                'fulfill_rate',
                                                'event_fulfill_rate',
                                                'unmet_demand',
                                                'cnt']],
                      on=['cnt'], suffixes=('', '_outsample'))

    # speedup cpu time & # of explored nodes
    ret_df['speedup_cpu_time'] = -1
    ret_df['speedup_node'] = -1
    ret_df['node'] = -1
    if output['model'] == 'co':
        ret_df['speedup_cpu_time'] = output['speedup_cpu_time']
        ret_df['speedup_node'] = output['speedup_node']
        ret_df['node'] = output['node']
    if output['model'] == 'mv':
        ret_df['node'] = output['node']
    if output['model'] == 'saa':
        pass
    return ret_df


def Construct_Parse_Output_Params():
    """
    under modifications!
    :return:
    """
    p = 'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system'

    # k: (rho, cv, kappa)
    with open(p + '/suffix_index.txt', 'r') as f:
        suffix_index = json.loads(f.readline())

    params = []
    for m, n in [(8, 8)]:
        for g in range(50):
            dir_path = p + f'/{m}{n}/graph{g}/output'
            for file in os.listdir(dir_path):
                k = re_find_k.match(file)[1]
                # if k == '0':
                rho_cv_kappa = suffix_index[k]
                file_path = dir_path + '/' + file
                param = {'path': file_path,
                         'm': m,
                         'n': n,
                         'rho': rho_cv_kappa['rho'],
                         'cv': rho_cv_kappa['cv'],
                         'kappa': rho_cv_kappa['kappa'],
                         'g': g}
                params.append(param)
    return params


if __name__ == '__main__':
    params = Construct_Parse_Output_Params()
    print(params)
    dfs = []
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        tasks = [executor.submit(Parse_Output, param) for param in params]
        for task in futures.as_completed(tasks):
            task_return = task.result()
            dfs.append(task_return)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv('D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/88/e_result_sensitivity.csv',
               index=False)
