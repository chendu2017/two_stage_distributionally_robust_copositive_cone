TOL = 1e-5


def Construct_A_Matrix(m, n, roads):
    """
    Construct the coefficient matrix of A = [a_1^T
                                            a_2^T
                                            ...
                                            a_M^T]
    :param m: # of warehouse locations
    :param n: # of demand points
    :param roads: a list of (i,j) pair
    :return: 2by2 list
    """
    import numpy as np
    r = len(roads)
    M, N = m + n + r, 2 * m + 2 * n + r
    A = np.zeros(shape=(M + 1, N + 1))  # 多一行一列，方便输入矩阵，后面删掉
    for row in range(1, M + 1):
        if row <= r:
            i, j = roads[row - 1]
            A[row, j] = 1
            A[row, n + i] = -1
            A[row, n + m + row] = 1

        if r < row <= r + n:
            j = row - r
            A[row, j] = 1
            A[row, j + m + n + r] = 1

        if r + n < row <= r + n + m:
            i = row - r - n
            A[row, n + i] = 1
            A[row, n + i + m + n + r] = 1

    return A[1:, 1:].tolist()


def Construct_b_vector(m, n, roads):
    r = len(roads)
    b = [0] * r + [1] * (m + n)
    return b


def isAllInteger(numbers):
    allIntegerFlag = all(map(isZeroOneInteger, numbers))
    return allIntegerFlag


def isZeroOneInteger(x):
    return abs(x - 1) <= TOL or abs(x) <= TOL