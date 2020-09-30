TOL = 1e-5


def isAllInteger(numbers):
    allIntegerFlag = all(map(isZeroOneInteger, numbers))
    return allIntegerFlag


def isZeroOneInteger(x):
    return abs(x - 1) <= TOL or abs(x) <= TOL