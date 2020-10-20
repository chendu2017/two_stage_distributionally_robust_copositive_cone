from concurrent import futures
import json


def Run_Single_Add_det_params_inputs(path):
    with open(path, 'r') as f_input:
        param = json.loads(f_input.readline())
    mu = param['e_param']['mu']
    param['det_param'] = {'mu': mu}
    with open(path, 'w') as f_output:
        f_output.write(json.dumps(param))


def Add_det_params_into_inputs():
    paths = []
    for g in range(50):
        for k in range(1, 29):
            paths.append('D:/[PAPER]NetworkDesign Distributionally Robust/'
                         f'numerical/balanced_system/88/graph{g}/input/input{k}.txt')
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        tasks = [executor.submit(Run_Single_Add_det_params_inputs, path) for path in paths]
        for task in futures.as_completed(tasks):
            pass


if __name__ == '__main__':
    Add_det_params_into_inputs()