from typing import List, Dict
import os
TOL = 1e-7


def isAllInteger(numbers):
    allIntegerFlag = all(map(isZeroOneInteger, numbers))
    return allIntegerFlag


def isZeroOneInteger(x):
    return abs(x - 1) <= TOL or abs(x) <= TOL


def Remove_Input(path):
    for m, n in [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:
        for g in range(20):
            for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                input_path = path + f'/{m}{n}/graph{g}/{mode}/input'
                file_lists = os.listdir(input_path)
                for file in file_lists:
                    os.remove(input_path + f'/{file}')


def Remove_Output(path):
    for m, n in [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:
        for g in range(20):
            for mode in ['equal_mean', 'non_equal_mean', 'non_equal_mean_mixture_gaussian']:
                output_path = path + f'/{m}{n}/graph{g}/{mode}/output'
                file_lists = os.listdir(output_path)
                for file in file_lists:
                    os.remove(output_path + f'/{file}')
