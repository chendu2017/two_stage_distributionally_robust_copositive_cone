from time import time
import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from co.Branch_Bound import BranchBound
from numerical_study.ns_utils import Calculate_Sampled_Cov_Matrix


class COModel(object):
    def __init__(self, m, n, f, h, mu_sampled, sigma_sampled, graph, mu_lb, mu_ub, sigma_lb, sigma_ub,
                 co_param=None, bootstrap=False, seed=int(time())):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.mu, self.sigma = mu_sampled, sigma_sampled
        self.graph = graph
        self.co_param = co_param
        self.bootstrap = False
        self.seed = seed
        if bootstrap:
            self.mu_lb, self.mu_ub, self.sigma_lb, self.sigma_ub = mu_lb, mu_ub, sigma_lb, sigma_ub
            self.bootstrap = True
        else:
            self.mu_lb, self.mu_ub, self.sigma_lb, self.sigma_ub = self.mu, self.mu, self.sigma, self.sigma
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]

        self.model = self._Build_Co_Model()

    def Solve_Co_Model(self) -> (Model, int):
        bb = BranchBound(self.model, self.mu, self.sigma, self.graph, self.co_param['bb_params'])
        bb.BB_Solve()
        self.model.dispose()
        return bb.best_model, bb.node_explored

    def _Build_Co_Model(self):
        mu, sigma = self.mu, self.sigma
        m, n, r = self.m, self.n, len(self.roads)
        f, h = self.f, self.h
        M, N = m + n + r, 2 * m + 2 * n + r
        A = self._Construct_A_Matrix()
        A_Mat = Matrix.dense(A)
        b = self._Construct_b_vector()

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
        M1 = xi, Xi,   psi^t
             phi, psi,   w  ]
        '''
        # no-need speedup variables
        Tau = COModel.variable('Tau', 1, Domain.unbounded())
        xi = COModel.variable('xi', n, Domain.unbounded())  # n by 1 vector
        Xi = COModel.variable('Xi', [n, n], Domain.unbounded())  # n by n matrix
        Psi = COModel.variable('Psi', [N, n], Domain.unbounded())
        Phi = COModel.variable('Phi', N, Domain.unbounded())  # N by 1 vector
        W = COModel.variable('W', [N, N], Domain.unbounded())

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

        if self.bootstrap:
            Xi_overline = COModel.variable('Xi_overline', [n, n], Domain.greaterThan(0.0))
            Xi_underline = COModel.variable('Xi_underline', [n, n], Domain.greaterThan(0.0))
            xi_overline = COModel.variable('xi_overline', n, Domain.greaterThan(0.0))
            xi_underline = COModel.variable('xi_underline', n, Domain.greaterThan(0.0))

        # -- Objective Function
        obj_1 = Expr.dot(f, Z)
        obj_2 = Expr.dot(h, I)
        obj_3 = Expr.dot(b, Alpha)
        obj_4 = Expr.dot(b, Beta)
        obj_5 = Expr.dot([1], Expr.add(Tau, a_M2))
        if self.bootstrap:
            obj_6 = Expr.sub(Expr.dot(self.mu_ub, xi_overline),
                             Expr.dot(self.mu_lb, xi_underline))
            obj_7 = Expr.sub(Expr.dot(self.sigma_ub, Xi_overline),
                             Expr.dot(self.sigma_lb, Xi_underline))
        else:
            obj_6 = Expr.dot([2 * mean for mean in mu], Expr.add(xi, b_M2))
            obj_7 = Expr.dot(sigma, Expr.add(Xi, e_M2))
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
        COModel.constraint('constr4', Expr.sub(Expr.mul(20000.0, Z), I), Domain.greaterThan(0.0))

        # Constraint 5: M1 is SDP
        COModel.constraint('constr5', Expr.vstack(Expr.hstack(Tau, xi.transpose(), Phi.transpose()),
                                                  Expr.hstack(xi, Xi, Psi.transpose()),
                                                  Expr.hstack(Phi, Psi, W)),
                           Domain.inPSDCone(1 + n + N))

        # Constraint 6: for bootstrap, there are two additional constraints
        if self.bootstrap:
            COModel.constraint('constr6_1', Expr.sub(Expr.sub(xi_overline, xi_underline),
                                                     Expr.mul(2, Expr.add(xi, b_M2))),
                               Domain.equalsTo(0))
            COModel.constraint('constr6_2', Expr.sub(Expr.sub(Xi_overline, Xi_underline),
                                                     Expr.add(Xi, e_M2)),
                               Domain.equalsTo(0))
        return COModel

    def _Construct_A_Matrix(self):
        """
        Construct the coefficient matrix of A = [a_1^T
                                                a_2^T
                                                ...
                                                a_M^T]
        :param roads: a list of (i,j) pair
        :return: 2by2 list
        """
        r = len(self.roads)
        m, n = self.m, self.n
        M, N = m + n + r, 2 * m + 2 * n + r
        A = np.zeros(shape=(M, N))
        for row in range(M):
            if row <= r - 1:
                i, j = self.roads[row]
                A[row, j] = 1
                A[row, n + i] = -1
                A[row, n + m + row] = 1

            if r - 1 < row <= r + n - 1:
                j = row - r
                A[row, j] = 1
                A[row, j + m + n + r] = 1

            if r + n - 1 < row <= r + n + m - 1:
                i = row - r - n
                A[row, n + i] = 1
                A[row, n + i + m + n + r] = 1

        return A.tolist()

    def _Construct_b_vector(self):
        r = len(self.roads)
        b = [0] * r + [1] * (self.m + self.n)
        return b


if __name__ == '__main__':
    from test_example.four_by_eight import *

    co_model = COModel(m, n, f, h, mu_sampled, sigma_sampled, graph, mu_lb, mu_ub, sigma_lb, sigma_ub,
                       co_param={'bb_params': {'find_init_z': 'v2',
                                               'select_branching_pos': 'v2'},
                                 'replicates': 10000,
                                 'CI': 10}, bootstrap=False)
    solved_co_model, node_explored = co_model.Solve_Co_Model()
    I = solved_co_model.getVariable('I').level()
    print(I)
    print('obj:', solved_co_model.primalObjValue())
    # times = []
    # K = 1
    # for i in range(K):
    #     start = time.perf_counter()
    #     solved_co_model = co_model.Solve_Co_Model()
    #     times.append(time.perf_counter() - start)
    # print('CPU Time (avg):', sum(times) / K, 'std:', )  # np.asarray(times).std())
# mosek -d MSK_IPAR_INFEAS_REPORT_AUTO MSK_ON infeas.lp -info rinfeas.lp
    sigma_sampled = sigma_sampled - cov_sampled + np.diag(np.diag(cov_sampled))
    mv_model = COModel(m, n, f, h, mu_sampled, sigma_sampled, graph, mu_lb, mu_ub, sigma_lb, sigma_ub,
                       co_param={'bb_params': {'find_init_z': 'v2',
                                               'select_branching_pos': 'v2'},
                                 'replicates': 10000,
                                 'CI': 10}, bootstrap=False)
    solved_mv_model, _ = mv_model.Solve_Co_Model()
    I = solved_mv_model.getVariable('I').level()
    print(I)
    print('obj:', solved_mv_model.primalObjValue())






