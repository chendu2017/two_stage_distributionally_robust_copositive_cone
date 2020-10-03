from copy import deepcopy

from Experiment import Experiment
import json
import numpy as np

if __name__ == '__main__':
    for k in range(2):
        with open(f'./numerical_study/balanced_system/test/input/input{k}.txt') as f:
            line = f.readline()
            e_params = json.loads(line)

            co_params = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                         'bb_params': {'find_init_z': 'v1',
                                       'select_branching_pos': 'v1'}}

            e = Experiment(e_params)
            m = e.m

            # run different model
            # co
            co_params = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                         'bb_params': {'find_init_z': 'v2',
                                       'select_branching_pos': 'v2'}}
            co_model = e.Run_Co_Model(co_params)
            sol = {'I': np.round(co_model.getVariable('I').level(), 2).tolist(),
                   'Z': np.round(co_model.getVariable('Z').level()).tolist()}
            co_simulation_in_sample = e.Simulate_in_Sample(sol)
            co_output = {'e_params': e_params,
                         'co_params': co_params,
                         'sol': sol,
                         'simulation_in_sample': co_simulation_in_sample}
            co_output = json.dumps(co_output) + '\n'

            # mv
            mv_params = deepcopy(co_params)
            mv_model = e.Run_MV_Model(mv_params)
            sol = {'I': np.round(mv_model.getVariable('I').level(), 2).tolist(),
                   'Z': np.round(mv_model.getVariable('Z').level()).tolist()}
            mv_simulation_in_sample = e.Simulate_in_Sample(sol)
            mv_output = {'e_params': e_params,
                         'mv_params': co_params,
                         'sol': sol,
                         'simulation_in_sample': mv_simulation_in_sample}
            mv_output = json.dumps(mv_output) + '\n'

            # saa
            saa_params = {}
            saa_model = e.Run_SAA_Model()
            sol = {'I': np.round([saa_model.getVarByName(f'I[{i}]').x for i in range(m)], 2).tolist(),
                   'Z': np.round([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)], 2).tolist()}
            saa_simulation_in_sample = e.Simulate_in_Sample(sol)
            saa_output = {'e_params': e_params,
                          'saa_params': saa_params,
                          'sol': sol,
                          'simulation_in_sample': saa_simulation_in_sample}
            saa_output = json.dumps(saa_output) + '\n'

            with open(f'numerical_study/balanced_system/test//output/outputs{k}.txt', 'w') as f_output:
                f_output.write(co_output)
                f_output.write(mv_output)
                f_output.write(saa_output)
