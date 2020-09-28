import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense, SolutionStatus


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
    r = len(roads)
    M, N = m + n + 2 * r, 2 * m + 2 * n + 2 * r
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
            A[row, j + M] = 1

        if r + n < row <= r + n + m:
            i = row - r - n

            A[row, n + i] = 1
            A[row, n + i + M] = 1

        # for s_{ij} slackness variables
        if r + n + m < row <= r + n + m + r:
            tiao = row - r - n - m

            A[row, n + m + tiao] = 1
            A[row, n + m + r + n + m + tiao] = 1
    return A[1:, 1:].tolist()


def Construct_b_vector(m, n, roads):
    r = len(roads)
    b = [0] * r + [1] * (m + n + r)
    return b


def BuildModel_Reduced(m, n, f, h, mu, sigma, graph):
    roads = [(i + 1, j + 1) for i in range(m) for j in range(n) if graph[i][j] == 1]
    r = len(roads)
    M, N = m + n + 2 * r, 2 * m + 2 * n + 2 * r
    A = Construct_A_Matrix(m, n, roads)
    A_Mat = Matrix.dense(A)
    b = Construct_b_vector(m, n, roads)

    # ---- build Mosek Model
    COModel = Model()

    # -- Decision Variable
    Z = COModel.variable('Z', m, Domain.inRange(0.0, 1.0))
    # Z = COModel.parameter(n)
    # Z.setValue([1]*n)
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
    obj_6 = Expr.dot([2 * mean for mean in mu], Expr.add(Xi, b_M2))
    obj_7 = Expr.dot(sigma, Expr.add(Eta, e_M2))
    COModel.objective(ObjectiveSense.Minimize, Expr.add([obj_1, obj_2, obj_5, obj_7]))
    # COModel.objective(ObjectiveSense.Minimize, Expr.add([obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7]))

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
    del _expr

    # Constraint 3
    _expr = Expr.mul(-2, Expr.add(Psi, d_M2))
    _expr_rhs = Matrix.sparse([[Matrix.eye(n)], [Matrix.sparse(N - n, n)]])
    COModel.constraint('constr3', Expr.sub(_expr, _expr_rhs), Domain.equalsTo(0))
    del _expr, _expr_rhs

    # Constraint 4: I <= M*Z
    COModel.constraint('constr4', Expr.sub(Expr.mul(10000.0, Z), I), Domain.greaterThan(0.0))

    # Constraint 5: M1 is SDP
    COModel.constraint('constr5', Expr.vstack(Expr.hstack(Tau, Xi.transpose(), Phi.transpose()),
                                              Expr.hstack(Xi, Eta, Psi.transpose()),
                                              Expr.hstack(Phi, Psi, W)),
                       Domain.inPSDCone(1 + n + N))
    return COModel


if __name__ == '__main__':
    from test_example.two_by_two import m, n, f, h, first_moment, second_moment, graph

    roads = [(i + 1, j + 1) for i in range(m) for j in range(n) if graph[i][j] == 1]
    b = Construct_b_vector(m, n, roads)
    A = Construct_A_Matrix(m, n, roads)

    model = BuildModel_Reduced(m, n, f, h, first_moment, second_moment, graph)
    model.writeTask('1.cbf')
    model.solve()
    print(model.getPrimalSolutionStatus())
    print(model.getVariable('I').level())
    if model.getPrimalSolutionStatus() == SolutionStatus.Optimal:
        I = model.getVariable('I').level()
        Z = model.getVariable('Z').level()
        # M1
        Alpha = model.getVariable('Alpha').level()
        Beta = model.getVariable('Beta').level()
        Theta = model.getVariable('Theta').level()
        Tau = model.getVariable('Tau').level()
        Xi = model.getVariable('Xi').level()
        Phi = model.getVariable('Phi').level()
        Phi = model.getVariable('Phi').level()
        Eta = model.getVariable('Eta').level()
        # M2
        a_M2 = model.getVariable('a_M2').level()
        b_M2 = model.getVariable('b_M2').level()
        c_M2 = model.getVariable('c_M2').level()
        d_M2 = model.getVariable('d_M2').level()
        e_M2 = model.getVariable('e_M2').level()
        f_M2 = model.getVariable('f_M2').level()

        print('objective value:', model.primalObjValue())
        print('obj_1=', sum(_*__ for _, __ in zip(f, Z)))
        print('obj_2=', sum(_*__ for _, __ in zip(h, I)))
        print('obj_3=', sum(_*__ for _, __ in zip(b, Alpha)))
        print('obj_4=', sum(_*__ for _, __ in zip(b, Beta)))
        print('obj_5=', sum(_*__ for _, __ in zip(Tau, a_M2)))
        print('obj_6=', sum(2*_*__ for _, __ in zip(Xi, b_M2)))
        print('obj_7=', sum(_*(__+___) for _, __, ___ in zip(sum(second_moment, []), Eta, e_M2)))
        print('I:', I)
        print('Z:', Z)
        print('Alpha:', Alpha)
        print('Beta:', Beta)
        print('Theta:', Theta)
        print('Tau:', Tau)
        print('Xi:', Xi)
        print('Phi:', Phi)
        print('Eta:', Eta)

        print('a_M2:', a_M2)
        print('b_M2:', b_M2)
        print('c_M2:', c_M2)
        print('d_M2:', d_M2)
        print('e_M2:', e_M2)
        print('f_M2:', f_M2)



# mosek -d MSK_IPAR_INFEAS_REPORT_AUTO MSK_ON infeas.lp -info rinfeas.lp
