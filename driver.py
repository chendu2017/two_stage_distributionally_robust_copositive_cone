from concurrent import futures
import gc
from Experiment import Experiment
import json
import numpy as np
from numerical_study.ns_utils import Write_Output, Chunks


def Run_CO(e, co_param, co_speedup_param):
    # co_speedup: only run it, when (rho, cv, kappa) is the default setting.
    if e.cv == 0.3 and e.rho == 0.3 and e.kappa == 1.0:
        co_model = e.Run_Co_Model(co_param)
        co_time, co_node = e.co_time, e.co_node
        sol = {'I': co_model.getVariable('I').level().tolist(),
               'Z': np.round(co_model.getVariable('Z').level()).tolist(),
               'obj': co_model.primalObjValue(),
               'h': np.matmul(co_model.getVariable('I').level(), e.h).tolist(),
               'f': np.matmul(co_model.getVariable('Z').level(), e.f).tolist(),
               }
        co_model.dispose()
        co_speedup_model = e.Run_Co_Model(co_speedup_param)
        co_speedup_time, co_speedup_node = e.co_time, e.co_node
        co_speedup_sol = {'I': co_speedup_model.getVariable('I').level().tolist(),
                          'Z': np.round(co_speedup_model.getVariable('Z').level()).tolist(),
                          'obj': co_speedup_model.primalObjValue(),
                          'h': np.matmul(co_speedup_model.getVariable('I').level(), e.h).tolist(),
                          'f': np.matmul(co_speedup_model.getVariable('Z').level(), e.f).tolist(),
                          }
        co_speedup_model.dispose()
    else:
        co_speedup_sol, co_speedup_time, co_speedup_node = {}, -1, -1
        co_model = e.Run_Co_Model(co_speedup_param)
        co_time, co_node = e.co_time, e.co_node
        sol = {'I': co_model.getVariable('I').level().tolist(),
               'Z': np.round(co_model.getVariable('Z').level()).tolist(),
               'obj': co_model.primalObjValue(),
               'h': np.matmul(co_model.getVariable('I').level(), e.h).tolist(),
               'f': np.matmul(co_model.getVariable('Z').level(), e.f).tolist(),
               }
        co_model.dispose()
    # simulation
    co_simulation = e.Simulate_Second_Stage(sol)
    co_output = {'model': 'co',
                 'sol': sol,
                 'speedup_sol': co_speedup_sol,
                 'cpu_time': co_time,
                 'node': co_node,
                 'speedup_cpu_time': co_speedup_time,
                 'speedup_node': co_speedup_node,
                 'simulation': co_simulation}
    return co_output


def Run_MV(e, mv_param):
    mv_model = e.Run_MV_Model(mv_param)
    mv_time, mv_node = e.mv_time, e.mv_node
    sol = {'I': mv_model.getVariable('I').level().tolist(),
           'Z': np.round(mv_model.getVariable('Z').level()).tolist(),
           'obj': mv_model.primalObjValue(),
           'h': np.matmul(mv_model.getVariable('I').level(), e.h).tolist(),
           'f': np.matmul(mv_model.getVariable('Z').level(), e.f).tolist()}
    mv_simulation, mv_simulation_outsample = e.Simulate_Second_Stage(sol)
    mv_output = {'model': 'mv',
                 'sol': sol,
                 'cpu_time': mv_time,
                 'node': mv_node,
                 'simulation': mv_simulation,
                 'simulation_outsample': mv_simulation_outsample}
    mv_model.dispose()
    return mv_output


def Run_SAA(e, saa_param):
    m = e.m
    saa_model = e.Run_SAA_Model(saa_param)
    saa_time = e.saa_time
    sol = {'I': [saa_model.getVarByName(f'I[{i}]').x for i in range(m)],
           'Z': np.round([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)]).tolist(),
           'obj': saa_model.ObjVal,
           'h': np.matmul([saa_model.getVarByName(f'I[{i}]').x for i in range(m)], e.h).tolist(),
           'f': np.matmul([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)], e.f).tolist()}
    saa_simulation, saa_simulation_outsample = e.Simulate_Second_Stage(sol)
    saa_output = {'model': 'saa',
                  'sol': sol,
                  'cpu_time': saa_time,
                  'simulation': saa_simulation,
                  'simulation_outsample': saa_simulation_outsample}
    saa_model.dispose()
    return saa_output


def Construct_Task_Params():
    task_params = []
    for m, n in [(6, 6)]:
        for _g in range(50):
            for k in range(1):
                task_param = {'dir_path': f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs/{m}{n}/graph{_g}',
                              'm': m,
                              'n': n,
                              'g': _g,
                              'k': k}
                task_params.append(task_param)
    return task_params


def Run_Single_Task(task_param):
    dir_path, m, n, g, k = \
        task_param['dir_path'], task_param['m'], task_param['n'], task_param['g'], task_param['k']

    # read input
    with open(dir_path + f'/input/input{k}.txt') as f_input:
        params = json.loads(f_input.readline())

    e_param, co_param, co_speedup_param, mv_param, saa_param \
        = params['e_param'], params['co_param'], params['co_speedup_param'], params['mv_param'], params['saa_param']

    e = Experiment(e_param)

    # co
    co_output = Run_CO(e, co_param, co_speedup_param)
    Write_Output(dir_path + '/output', co_output, k)
    print(f'----{(m,n)}----graph{g}----{k}----CO----' + 'Done----')

    # mv
    mv_output = Run_MV(e, mv_param)
    Write_Output(dir_path + '/output', mv_output, k)
    print(f'----{(m, n)}----graph{g}----{k}----MV----' + 'Done----')

    # saa
    saa_output = Run_SAA(e, saa_param)
    Write_Output(dir_path + '/output', saa_output, k)
    print(f'----{(m, n)}----graph{g}----{k}----SAA----' + 'Donse----')

    return f'----{(m,n)}----graph{g}----{k}----' + 'DoneDoneDoneDone----'


if __name__ == '__main__':
    task_params = Construct_Task_Params()

    try:
        for task_params in Chunks(task_params, 50):
            print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
            with futures.ProcessPoolExecutor(max_workers=2) as executor:
                tasks = [executor.submit(Run_Single_Task, task_param) for task_param in task_params]
                for task in futures.as_completed(tasks):
                    task_return = task.result()
                    print(task_return)
    except Exception as e:
        print(e)




