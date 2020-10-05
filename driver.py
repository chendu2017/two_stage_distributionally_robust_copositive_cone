import gc
from Experiment import Experiment
import json
import numpy as np
from memory_profiler import profile


@profile
def Run_CO(e, co_params, co_speedup_params):
    print('-------co-------')
    co_model = e.Run_Co_Model(co_params)
    co_time = e.co_time
    sol = {'I': co_model.getVariable('I').level().tolist(),
           'Z': np.round(co_model.getVariable('Z').level()).tolist(),
           'obj': co_model.primalObjValue(),
           'h': np.matmul(co_model.getVariable('I').level(), e.h).tolist(),
           'f': np.matmul(co_model.getVariable('Z').level(), e.f).tolist()
           }
    # co_speedup
    print('-------co speedup-------')
    e.Run_Co_Model(co_speedup_params)
    co_speedup_time = e.co_time
    # simulation
    co_simulation_in_sample = e.Simulate_in_Sample(sol)
    co_simulation_out_sample = e.Simulate_out_Sample(sol)
    co_output = {'model': 'co',
                 'sol': sol,
                 'cpu_time': co_time,
                 'speedup_cpu_time': co_speedup_time,
                 'simulation_in_sample': co_simulation_in_sample,
                 'simulation_out_sample': co_simulation_out_sample}
    return co_output


def Run_MV(e, mv_params):
    print('-------mv-------')
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
    return mv_output


def Run_SAA(e, saa_params):
    print('-------saa-------')
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
    return saa_output


def Run(params):
    e_params = params['e_params']
    co_params = params['co_params']
    co_speedup_params = params['co_speedup_params']
    mv_params = params['mv_params']
    saa_params = params['saa_params']

    e = Experiment(e_params)

    # run different model
    co_output = Run_CO(e, co_params, co_speedup_params)
    mv_output = Run_MV(e, mv_params)
    saa_output = Run_SAA(e, saa_params)

    output = {'params': params,
              'co_output': co_output,
              'mv_output': mv_output,
              'saa_output': saa_output}
    return output


if __name__ == '__main__':
    for m, n in [(4, 4)]:
        for _g in range(20):
            for k in range(36):
                print('--------------', 'graph:', _g, 'input-k:', k, '--------------')

                # equal_mean
                print('---------equal mean----------')
                with open(f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/equal_mean/input/input{k}.txt',) as f_input:
                    params = json.loads(f_input.readline())
                    output = json.dumps(Run(params))
                with open(f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/equal_mean/output/output{k}.txt', 'w') as f_output:
                    f_output.write(output)

                gc.collect()
                # # non-equal mean
                # print('---------non equal mean----------')
                # with open(f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean/input/input{k}.txt') as f_input:
                #     params = json.loads(f_input.readline())
                #     output = json.dumps(Run(params))
                # with open(f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean/output/output{k}.txt', 'w') as f_output:
                #     f_output.write(output)
                #
                # # non-equal mean - mixture gaussian
                # print('---------non equal mean mixture gaussian----------')
                # with open(
                #         f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean_mixture_gaussian/input/input{k}.txt') as f_input:
                #     params = json.loads(f_input.readline())
                #     output = json.dumps(Run(params))
                # with open(
                #         f'D:/[PAPER]NetworkDesign Distributionally Robust/numerical/balanced_system/{m}{n}/graph{_g}/non_equal_mean_mixture_gaussian/output/output{k}.txt',
                #         'w') as f_output:
                #     f_output.write(output)




