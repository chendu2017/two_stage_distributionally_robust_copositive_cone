import json
import numpy as np
from numerical_study.ns_utils import Generate_d_rs_out_sample, Generate_d_rs
from concurrent import futures
from simulator.Simulator import Simulator


def Run_Single_Add_d_rs_outsample_into_inputs(path):
    with open(path, 'r') as f_input:
        param = json.loads(f_input.readline())
    e_param = param['e_param']
    mu, sigma, rho, cv = e_param['mu'], e_param['sigma'], e_param['rho'], e_param['cv']
    cov_mat = sigma - np.outer(mu, mu)
    d_rs_out_sample = Generate_d_rs_out_sample(mu, cov_mat, cv, 1000)
    d_rs_out_sample = {f'{k}': d_r for k, d_r in enumerate(d_rs_out_sample)}
    param['e_param']['d_rs_outsample'] = d_rs_out_sample
    with open(path, 'w') as f_output:
        f_output.write(json.dumps(param))


def Run_Single_Add_d_rs_insample_into_inputs(path):
    with open(path, 'r') as f_input:
        param = json.loads(f_input.readline())
    e_param = param['e_param']
    mu, sigma, rho, cv = e_param['mu'], e_param['sigma'], e_param['rho'], e_param['cv']
    cov_mat = sigma - np.outer(mu, mu)
    d_rs = Generate_d_rs(mu, cov_mat, 1000)
    d_rs = {f'{k}': d_r for k, d_r in enumerate(d_rs)}
    param['e_param']['d_rs'] = d_rs
    with open(path, 'w') as f_output:
        f_output.write(json.dumps(param))


def Add_d_rs_outsample_into_inputs():
    paths = []
    for g in range(50):
        for k in range(1, 29):
            paths.append('D:/[PAPER]NetworkDesign Distributionally Robust/'
                         f'numerical/balanced_system/.new_inputs/88/graph{g}/input/input{k}.txt')
    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(Run_Single_Add_d_rs_outsample_into_inputs, path) for path in paths]
        for task in futures.as_completed(tasks):
            pass


def Add_d_rs_insample_into_inputs():
    paths = []
    for g in range(50):
        for k in range(1, 29):
            paths.append('D:/[PAPER]NetworkDesign Distributionally Robust/'
                         f'numerical/balanced_system/.new_inputs/88/graph{g}/input/input{k}.txt')
    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(Run_Single_Add_d_rs_insample_into_inputs, path) for path in paths]
        for task in futures.as_completed(tasks):
            pass


def Construct_Task_Params():
    params = []
    for m, n in [(8, 8)]:
        for g in range(50):
            for k in range(1, 29):
                param = {'m': m,
                         'n': n,
                         'k': k,
                         'g': g,
                         'path': 'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/.new_inputs'}
                params.append(param)
    return params


def Run_Single_Outsample_Simulation(param):
    path = param['path']
    m, n, g, k = param['m'], param['n'], param['g'], param['k']

    # get network structure
    input_file_path = path + f'/{m}{n}/graph{g}/input/input{k}.txt'
    with open(input_file_path, 'r') as f_input:
        input_param = json.loads(f_input.readline())
        graph = input_param['e_param']['graph']
        d_rs_outsample = input_param['e_param']['d_rs_outsample']

    # for each model, run
    for model in ['co', 'mv', 'saa']:
        output_file_path = path + f'/{m}{n}/graph{g}/output/output{k}_{model}.txt'
        with open(output_file_path, 'r') as f_output:
            output_ = json.loads(f_output.readline())
            sol = {'I': output_['sol']['I'],
                   'Z': output_['sol']['Z']}

        # simulate
        simulator = Simulator(m, n, graph)
        simulator.setSol(sol)
        simulator.setDemand_Realizations(d_rs_outsample)
        res = simulator.Run_Simulations()
        output_['simulation_outsample'] = res
        with open(output_file_path, 'w') as f_output:
            f_output.write(json.dumps(output_))
    return f'----({m}, {n})----graph{g}----outsample----DONEDONEDONE'


def Run_Single_Insample_Simulation(param):
    path = param['path']
    m, n, g, k = param['m'], param['n'], param['g'], param['k']

    # get network structure
    input_file_path = path + f'/{m}{n}/graph{g}/input/input{k}.txt'
    with open(input_file_path, 'r') as f_input:
        input_param = json.loads(f_input.readline())
        graph = input_param['e_param']['graph']
        d_rs = input_param['e_param']['d_rs']

    # for each model, run
    for model in ['co', 'mv', 'saa']:
        output_file_path = path + f'/{m}{n}/graph{g}/output/output{k}_{model}.txt'
        with open(output_file_path, 'r') as f_output:
            output_ = json.loads(f_output.readline())
            sol = {'I': output_['sol']['I'],
                   'Z': output_['sol']['Z']}

        # simulate
        simulator = Simulator(m, n, graph)
        simulator.setSol(sol)
        simulator.setDemand_Realizations(d_rs)
        res = simulator.Run_Simulations()
        output_['simulation'] = res
        with open(output_file_path, 'w') as f_output:
            f_output.write(json.dumps(output_))
    return f'----({m}, {n})----graph{g}----insample----DONEDONEDONE'


if __name__ == '__main__':
    Add_d_rs_outsample_into_inputs()
    # Add_d_rs_insample_into_inputs()

    task_params = Construct_Task_Params()

    # print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
    # with futures.ProcessPoolExecutor(max_workers=4) as executor:
    #     tasks = [executor.submit(Run_Single_Insample_Simulation, task_param) for task_param in task_params]
    #     for task in futures.as_completed(tasks):
    #         task_return = task.result()
    #         print(task_return)

    print('\n\n\n\n\n NEW EXECUTOR \n\n\n\n\n')
    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(Run_Single_Outsample_Simulation, task_param) for task_param in task_params]
        for task in futures.as_completed(tasks):
            task_return = task.result()
            print(task_return)
