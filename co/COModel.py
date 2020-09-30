import numpy as np
from typing import Dict

from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense, SolutionStatus
from co.Branch_Bound import BranchBound


def Solve_Co_Model(model: Model):
    bb = BranchBound(model)
    bb.BB_Solve()
    return bb.best_node.model


def Build_Co_Model(m, n, f, h, mu, sigma, graph, co_params=None):
    roads = [(i + 1, j + 1) for i in range(m) for j in range(n) if graph[i][j] == 1]
    r = len(roads)
    M, N = m + n + r, 2 * m + 2 * n + r
    A = __Construct_A_Matrix(m, n, roads)
    A_Mat = Matrix.dense(A)
    b = __Construct_b_vector(m, n, roads)

    # ---- build Mosek Model
    COModel = Model()

    # -- Decision Variable
    Z = COModel.variable('Z', m, Domain.inRange(0.0, 1.0))
    I = COModel.variable('I', m, Domain.greaterThan(0.0))
    Alpha = COModel.variable('Alpha', M, Domain.unbounded())  # M by 1 vector
    Beta = COModel.variable('Beta', M, Domain.unbounded())  # M by 1 vector
    Theta = COModel.variable('Theta', N, Domain.unbounded())  # N by 1 vector
    # M1_matrix related decision variables
    '''
        [tau, xi^T, phi^T
    M1 = xi, eta,   psi^t
         phi, psi,   w  ]
    '''
    Tau = COModel.variable('Tau', 1, Domain.unbounded())  # scalar
    Xi = COModel.variable('Xi', n, Domain.unbounded())  # n by 1 vector
    Phi = COModel.variable('Phi', N, Domain.unbounded())  # N by 1 vector
    Eta = COModel.variable('Eta', [n, n], Domain.unbounded())  # n by n matrix
    Psi = COModel.variable('Psi', [N, n], Domain.unbounded())
    W = COModel.variable('W', [N, N], Domain.unbounded())  # N by N matrix
    # M2 matrix decision variables
    '''
        [a, b^T, c^T
    M2 = b, e,   d^t
         c, d,   f  ]
    '''
    a_M2 = COModel.variable('a_M2', 1, Domain.greaterThan(0.0))
    b_M2 = COModel.variable('b_M2', n, Domain.greaterThan(0.0))
    c_M2 = COModel.variable('c_M2', N, Domain.greaterThan(0.0))
    e_M2 = COModel.variable('e_M2', [n, n], Domain.greaterThan(0.0))
    d_M2 = COModel.variable('d_M2', [N, n], Domain.greaterThan(0.0))
    f_M2 = COModel.variable('f_M2', [N, N], Domain.greaterThan(0.0))

    # -- Objective Function
    obj_1 = Expr.dot(f, Z)
    obj_2 = Expr.dot(h, I)
    obj_3 = Expr.dot(b, Alpha)
    obj_4 = Expr.dot(b, Beta)
    obj_5 = Expr.dot([1], Expr.add(Tau, a_M2))
    obj_6 = Expr.dot([2*mean for mean in mu], Expr.add(Xi, b_M2))
    obj_7 = Expr.dot(sigma, Expr.add(Eta, e_M2))
    COModel.objective(ObjectiveSense.Minimize, Expr.add([obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7]))

    # Constraint 1
    _expr = Expr.sub(Expr.mul(A_Mat.transpose(), Alpha), Theta)
    _expr = Expr.sub(_expr, Expr.mul(2, Expr.add(Phi, c_M2)))
    _expr_rhs = Expr.vstack(Expr.constTerm([0.0] * n), Expr.mul(-1, I), Expr.constTerm([0.0] * M))
    COModel.constraint('constr1', Expr.sub(_expr, _expr_rhs), Domain.equalsTo(0.0))
    del _expr, _expr_rhs

    # Constraint 2
    _first_term = Expr.add([Expr.mul(Beta.index(row), np.outer(A[row], A[row]).tolist()) for row in range(M)])
    _second_term = Expr.add([Expr.mul(Theta.index(k), Matrix.sparse(N, N, [k], [k], [1]))
                             for k in range(N)])
    _third_term = Expr.add(W, f_M2)
    _expr = Expr.sub(Expr.add(_first_term, _second_term), _third_term)
    COModel.constraint('constr2', _expr, Domain.equalsTo(0.0))
    del _expr, _first_term, _second_term, _third_term

    # Constraint 3
    _expr = Expr.mul(-2, Expr.add(Psi, d_M2))
    _expr_rhs = Matrix.sparse([[Matrix.eye(n)], [Matrix.sparse(N - n, n)]])
    COModel.constraint('constr3', Expr.sub(_expr, _expr_rhs), Domain.equalsTo(0))
    del _expr, _expr_rhs

    # Constraint 4: I <= M*Z
    COModel.constraint('constr4', Expr.sub(Expr.mul(2000.0, Z), I), Domain.greaterThan(0.0))

    # Constraint 5: M1 is SDP
    COModel.constraint('constr5', Expr.vstack(Expr.hstack(Tau, Xi.transpose(), Phi.transpose()),
                                              Expr.hstack(Xi, Eta, Psi.transpose()),
                                              Expr.hstack(Phi, Psi, W)),
                       Domain.inPSDCone(1 + n + N))

    return COModel


def __Construct_A_Matrix(m, n, roads):
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


def __Construct_b_vector(m, n, roads):
    r = len(roads)
    b = [0] * r + [1] * (m + n)
    return b


if __name__ == '__main__':
    from test_example.eight_by_eight import m, n, f, h, first_moment, second_moment, graph
    import sys
    roads = [(i + 1, j + 1) for i in range(m) for j in range(n) if graph[i][j] == 1]
    b = __Construct_b_vector(m, n, roads)
    A = __Construct_A_Matrix(m, n, roads)

    model = Build_Co_Model(m, n, f, h, first_moment, second_moment, graph)

# mosek -d MSK_IPAR_INFEAS_REPORT_AUTO MSK_ON infeas.lp -info rinfeas.lp
