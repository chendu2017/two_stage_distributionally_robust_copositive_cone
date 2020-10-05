from typing import List, Dict
TOL = 1e-7


def isAllInteger(numbers):
    allIntegerFlag = all(map(isZeroOneInteger, numbers))
    return allIntegerFlag


def isZeroOneInteger(x):
    return abs(x - 1) <= TOL or abs(x) <= TOL


def ConstructInitZ(m: int, locations: List[int]) -> Dict[int, int]:
    init_z = {}
    for i in range(m):
        if i in locations:
            init_z[i] = 1
        else:
            init_z[i] = 0
    return init_z
