from concurrent import futures
import gc
from Experiment import Experiment
import json
import numpy as np
from numerical_study.ns_utils import Write_Output, Chunks


def Run_CO(e, co_params, co_speedup_params):
    co_model = e.Run_Co_Model(co_params)
    co_time = e.co_time
    sol = {'I': co_model.getVariable('I').level().tolist(),
           'Z': np.round(co_model.getVariable('Z').level()).tolist(),
           'obj': co_model.primalObjValue(),
           'h': np.matmul(co_model.getVariable('I').level(), e.h).tolist(),
           'f': np.matmul(co_model.getVariable('Z').level(), e.f).tolist(),
           }
    co_model.dispose()
    # co_speedup
    co_speedup_model = e.Run_Co_Model(co_speedup_params)
    co_speedup_time = e.co_time
    co_speedup_sol = {'I': co_speedup_model.getVariable('I').level().tolist(),
                      'Z': np.round(co_speedup_model.getVariable('Z').level()).tolist(),
                      'obj': co_speedup_model.primalObjValue(),
                      'h': np.matmul(co_speedup_model.getVariable('I').level(), e.h).tolist(),
                      'f': np.matmul(co_speedup_model.getVariable('Z').level(), e.f).tolist(),
                      }
    co_speedup_model.dispose()
    # simulation
    co_simulation_in_sample = e.Simulate_in_Sample(sol)
    co_simulation_out_sample = e.Simulate_out_Sample(sol)
    co_output = {'model': 'co',
                 'sol': sol,
                 'speedup_sol': co_speedup_sol,
                 'cpu_time': co_time,
                 'speedup_cpu_time': co_speedup_time,
                 'simulation_in_sample': co_simulation_in_sample,
                 'simulation_out_sample': co_simulation_out_sample}
    return co_output


def Run_MV(e, mv_params):
    mv_model = e.Run_MV_Model(mv_params)
    mv_time = e.mv_time
    sol = {'I': mv_model.getVariable('I').level().tolist(),
           'Z': np.round(mv_model.getVariable('Z').level()).tolist(),
           'obj': mv_model.primalObjValue(),
           'h': np.matmul(mv_model.getVariable('I').level(), e.h).tolist(),
           'f': np.matmul(mv_model.getVariable('Z').level(), e.f).tolist()}
    mv_simulation_in_sample = e.Simulate_in_Sample(sol)
    mv_simulation_out_sample = e.Simulate_out_Sample(sol)
    mv_output = {'model': 'mv',
                 'sol': sol,
                 'cpu_time': mv_time,
                 'simulation_in_sample': mv_simulation_in_sample,
                 'simulation_out_sample': mv_simulation_out_sample}
    mv_model.dispose()
    return mv_output


def Run_SAA(e, saa_params):
    m = e.m
    saa_model = e.Run_SAA_Model(saa_params)
    saa_time = e.saa_time
    sol = {'I': [saa_model.getVarByName(f'I[{i}]').x for i in range(m)],
           'Z': np.round([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)]).tolist(),
           'obj': saa_model.ObjVal,
           'h': np.matmul([saa_model.getVarByName(f'I[{i}]').x for i in range(m)], e.h).tolist(),
           'f': np.matmul([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)], e.f).tolist()}
    saa_simulation_in_sample = e.Simulate_in_Sample(sol)
    saa_simulation_out_sample = e.Simulate_out_Sample(sol)
    saa_output = {'model': 'saa',
                  'sol': sol,
                  'cpu_time': saa_time,
                  'simulation_in_sample': saa_simulation_in_sample,
                  'simulation_out_sample': saa_simulation_out_sample}
    saa_model.dispose()
    return saa_output


def Construct_Task_Params():
    task_params = []
    for m, n in [(4, 4)]:
        for _g in range(20):
            for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                for k in range(9):
                    task_param = {'dir_path': f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/{mode}',
                                  'm': m,
                                  'n': n,
                                  'g': _g,
                                  'k': k,
                                  'mode': mode}
                    task_params.append(task_param)
    return task_params


def Run_Single_Task(task_param):
    dir_path, m, n, g, k, mode = \
        task_param['dir_path'], task_param['m'], task_param['n'], task_param['g'], task_param['k'], task_param['mode']

    # read input
    with open(dir_path + f'/input/input{k}.txt') as f_input:
        params = json.loads(f_input.readline())
        e_params, co_params, co_speedup_params, mv_params, saa_params \
            = params['e_params'], params['co_params'], params['co_speedup_params'], params['mv_params'], params['saa_params']

    e = Experiment(e_params)
    # co
    co_output = Run_CO(e, co_params, co_speedup_params)
    Write_Output(dir_path + '/output', co_output, k)
    print(f'----{(m,n)}----graph{g}----{mode}----{k}----CO----' + 'Done----')
    # mv
    mv_output = Run_MV(e, mv_params)
    Write_Output(dir_path + '/output', mv_output, k)
    print(f'----{(m, n)}----graph{g}----{mode}----{k}----MV----' + 'Done----')
    # saa
    saa_output = Run_SAA(e, saa_params)
    Write_Output(dir_path + '/output', saa_output, k)
    print(f'----{(m, n)}----graph{g}----{mode}----{k}----SAA----' + 'Done----')

    return f'----{(m,n)}----graph{g}----{mode}----{k}----' + 'DoneDoneDoneDone----'


if __name__ == '__main__':
    task_params = Construct_Task_Params()

    for task_params in Chunks(task_params, 20):
        print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
        with futures.ProcessPoolExecutor(max_workers=3) as executor:
            tasks = [executor.submit(Run_Single_Task, task_param) for task_param in task_params]
            for task in futures.as_completed(tasks):
                task_return = task.result()
                print(task_return)


# TODO:
# 1. experiment: sampled-mean -> real mean
# 2. mixture gaussian should be moved to extension



