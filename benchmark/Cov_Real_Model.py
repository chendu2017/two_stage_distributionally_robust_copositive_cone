from mosek.fusion import Model, Domain, Expr, ObjectiveSense
from itertools import product
import numpy as np
from gurobipy.gurobipy import Model as grbModel, GurobiError, GRB
from gurobipy.gurobipy import GRB, quicksum
from co.Branch_Bound import BranchBound
from mosek.fusion import Model as mskModel
from mosek.fusion import Expr, Domain
from gurobipy import GurobiError
from numerical_study.ns_utils import Generate_d_rs, Calculate_Sampled_Sigma_Matrix
from deprecated import deprecated
from mosek.fusion import ProblemStatus

INF = float('inf')
TOL = 1e-4
BIG_M = 20000


@deprecated(reason='use a subset of extreme point to approximate')
class BranchBoundSDP(BranchBound):
    def __init__(self, model: mskModel, mu, sigma, graph, bb_params):
        super().__init__(model, mu, sigma, graph, bb_params)

    def BB_Solve(self):
        super().BB_Solve(solve_func=self._Solve_by_Constraint_Generation)

    def _Solve_by_Constraint_Generation(self, master_model: mskModel) -> mskModel:
        while True:
            master_model.solve()
            # master_model.getTask().__del__()
            if master_model.getProblemStatus() == ProblemStatus.PrimalInfeasible:
                return master_model
            # get mskModel's variables
            Xi = master_model.getVariable('Xi')
            xi = master_model.getVariable('xi')
            I = master_model.getVariable('I')
            Tau = master_model.getVariable('Tau')
            # get mskModel's optimal values
            Xi_star = np.asarray(Xi.level()).reshape(self.n, self.n)
            xi_star = xi.level()
            I_star = I.level()
            Tau_star = Tau.level()
            print('I_star:', I_star)

            # build sub_model with grbModel
            sub_model = self._Build_CG_Sub_Model(Xi_star, xi_star, I_star, Tau_star)
            try:
                sub_model.optimize()
            except GurobiError as e:
                sub_model.params.NonConvex = 2
                sub_model.optimize()

            if sub_model.ObjVal <= -TOL:
                vars = sub_model.getVars()
                v_star = [v.x for v in vars[self.n: self.n + self.m]]
                u_hat_star = [u_hat.x for u_hat in vars[self.n + self.m:]]
                print('add constraint for (v,u_hat):', v_star, u_hat_star)
                master_model.constraint(Expr.stack([[Xi, Expr.mul(1 / 2, Expr.sub(xi, u_hat_star))],
                                                    [Expr.mul(1 / 2, Expr.sub(xi.transpose(), [u_hat_star])),
                                                     Expr.add(Expr.dot(I, v_star), Tau)]]
                                                   ), Domain.inPSDCone(self.n + 1))
                sub_model.__del__()
            else:
                break
        return master_model

    def _Build_CG_Sub_Model(self, Xi_star, xi_star, I_star, Tau_star) -> grbModel:
        sub_model = grbModel('subModel')
        sub_model.setParam('OutputFlag', 1)
        sub_model.modelSense = GRB.MINIMIZE

        d = sub_model.addMVar(self.n, vtype=GRB.CONTINUOUS, lb=-200, name='d')
        v = sub_model.addMVar(self.m, vtype=GRB.BINARY, name='v')
        u_hat = sub_model.addMVar(self.n, vtype=GRB.BINARY, name='u_hat')

        obj_1 = d @ Xi_star @ d
        obj_2 = d @ (xi_star - u_hat)
        obj_3 = v @ I_star
        obj_4 = Tau_star
        # obj_5 = BIG_M * (v @ np.diag([1] * self.m) @ v - v @ np.ones(self.m))  # to make Q SDP (diagonal dominant)
        # obj_6 = BIG_M * (u_hat @ np.diag([1] * self.n) @ u_hat - u_hat @ np.ones(self.n))  # to make Q become SDP
        objFunc = obj_1 + obj_2 + obj_3 + obj_4  # + obj_5 + obj_6
        sub_model.setObjective(objFunc)

        # 约束1: v_i = u_hat_j \forall (i,j) in roads
        # 约束2: v_i >= u_hat_j \for all (i,j) not in roads
        sub_model.addConstrs(v[i] == u_hat[j] for (i, j) in self.roads)
        sub_model.addConstrs(v[i] >= u_hat[j] for i in range(self.m) for j in range(self.n))
        sub_model.addConstr(objFunc >= -200000)

        # model parameters
        sub_model.update()
        sub_model.setParam('NodeLimit', int(np.log(self.m + self.n) * 5000 + 5000))
        sub_model.setParam('timeLimit', 300)
        sub_model.setParam('MIPFocus', 2)
        sub_model.setParam('Threads', 6)
        return sub_model


class CovRealModel(object):
    def __init__(self, m, n, f, h, mu_sampled, sigma_sampled, graph, mu_lb, mu_ub, sigma_lb, sigma_ub,
                 cov_real_param=None, bootstrap=False, ):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.mu, self.sigma = mu_sampled, sigma_sampled
        self.graph = graph
        self.cov_real_param = cov_real_param
        self.bootstrap = False
        if bootstrap:
            self.mu_lb, self.mu_ub, self.sigma_lb, self.sigma_ub = mu_lb, mu_ub, sigma_lb, sigma_ub
            self.bootstrap = True
        else:
            self.mu_lb, self.mu_ub, self.sigma_lb, self.sigma_ub = self.mu, self.mu, self.sigma, self.sigma
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]

        self.model = self.Build_Cov_Real_Model()

    def Build_Cov_Real_Model(self) -> Model:
        mu, sigma = self.mu, self.sigma
        mu_ub, mu_lb, sigma_ub, sigma_lb = self.mu_ub, self.mu_lb, self.sigma_ub, self.sigma_lb
        m, n, r = self.m, self.n, len(self.roads)
        f, h = self.f, self.h

        # model
        CovRealModel = Model()
        # -- Decision Variable
        Z = CovRealModel.variable('Z', m, Domain.inRange(0.0, 1.0))
        I = CovRealModel.variable('I', m, Domain.greaterThan(0.0))
        Tau = CovRealModel.variable('Tau', 1, Domain.greaterThan(0.0))  # 1 by 1 vector
        xi = CovRealModel.variable('xi', n, Domain.greaterThan(0.0))  # n by 1 vector
        Xi = CovRealModel.variable('Xi', [n, n], Domain.inPSDCone(n))  # n by n matrix

        if self.bootstrap:
            Xi_overline = CovRealModel.variable('Xi_overline', [n, n], Domain.greaterThan(0.0))
            Xi_underline = CovRealModel.variable('Xi_underline', [n, n], Domain.greaterThan(0.0))
            xi_overline = CovRealModel.variable('xi_overline', n, Domain.greaterThan(0.0))
            xi_underline = CovRealModel.variable('xi_underline', n, Domain.greaterThan(0.0))

        # objective
        obj_1 = Expr.dot(f, Z)
        obj_2 = Expr.dot(h, I)
        obj_3 = Expr.dot([1], Tau)
        if self.bootstrap:
            obj_4 = Expr.sub(Expr.dot(mu_ub, xi_overline),
                             Expr.dot(mu_lb, xi_underline))
            obj_5 = Expr.sub(Expr.dot(sigma_ub, Xi_overline),
                             Expr.dot(sigma_lb, Xi_underline))
        else:
            obj_4 = Expr.dot(mu.tolist(), xi)
            obj_5 = Expr.dot(sigma.tolist(), Xi)
        CovRealModel.objective(ObjectiveSense.Minimize, Expr.add([obj_1, obj_2, obj_3, obj_4, obj_5]))

        # Constraint 1: SDP matrix for each extreme points.
        # two extreme points are (1, 1) and (0, 0)
        extreme_points = self._Find_Extreme_Points_by_Solving_MIP()
        # extreme_points = [([1] * self.m, [1] * self.n), ([0] * self.m, [0]*self.n)]
        for v, hat_u in extreme_points:
            v, hat_u = list(v), list(hat_u)
            CovRealModel.constraint('constr_SDP', Expr.stack([[Xi, Expr.mul(1 / 2, Expr.sub(xi, hat_u))],
                                                              [Expr.mul(1 / 2, Expr.sub(xi.transpose(), [hat_u])),
                                                               Expr.add(Expr.dot(I, v), Tau)]]
                                                             ), Domain.inPSDCone(n + 1))

        # Constraint 2: if bootstrap, ub - lb - mean = 0
        if self.bootstrap:
            CovRealModel.constraint('constr2_1', Expr.sub(xi_overline, Expr.add(xi_underline, xi)),
                                    Domain.equalsTo(0))
            CovRealModel.constraint('constr2_2', Expr.sub(Xi_overline, Expr.add(Xi_underline, Xi)),
                                    Domain.equalsTo(0))

        # Constraint 3: I <= M*Z
        CovRealModel.constraint('constr3', Expr.sub(Expr.mul(50000, Z), I), Domain.greaterThan(0.0))

        # Constraint 4: all stores should be covered
        # graph = np.asarray(self.graph)
        # print(graph)
        # for j in range(self.n):
        #     CovRealModel.constraint('constr4_%d' % j, Expr.dot(graph[:, j].tolist(), Z), Domain.greaterThan(1.0-TOL))
        return CovRealModel

    def Solve_Cov_Real_Model(self):
        bb = BranchBound(self.model, self.mu, self.sigma, self.graph, self.f, self.h, self.cov_real_param['bb_params'])
        bb.BB_Solve()
        self.model.dispose()
        return bb.best_model, bb.node_explored

    @deprecated(reason='not necessary to find all extreme points')
    def _Find_All_Extreme_Points(self):
        extreme_points = []
        points = product([0, 1], repeat=self.n + self.m)
        for point in points:
            v = point[:self.m]
            u_hat = point[self.m: self.m + self.n]
            if self._Is_Extreme_Point(v, u_hat, self.graph):
                extreme_points.append([v, u_hat])
        return extreme_points

    def _Find_Extreme_Points_by_Solving_MIP(self):
        extreme_points = []

        _model = grbModel('subModel')
        _model.setParam('OutputFlag', 1)
        _model.modelSense = GRB.MAXIMIZE

        # Variables
        v = _model.addVars(self.m, vtype=GRB.BINARY, obj=1, name='v')
        u_hat = _model.addVars(self.n, vtype=GRB.BINARY, obj=1, name='u_hat')

        # Constraints
        # iss, js = np.where(np.asarray(graph) == 1)
        # for i, j in zip(iss, js):
        #     _model.addConstr(v[i] == u_hat[j])
        # iss_0, js_0 = np.where(np.asarray(graph) == 0)
        # for i, j in zip(iss_0, js_0):
        #     _model.addConstr(v[i] >= u_hat[j])
        for (i, j) in self.roads:
            _model.addConstr(v[i] == u_hat[j])

        # Solve
        _model.setParam('OutputFlag', 0)
        _model.setParam('MIPFocus', 1)  # focus on finding feasible solutions
        _model.setParam('PoolSearchMode', 2)  # search for more solutions, but in a non-systematic way
        _model.setParam('PoolSolutions', 100)
        _model.optimize()

        # get some feasible solutions
        solution_number = _model.SolCount
        solution_number_selected = int(min(solution_number, np.log(solution_number) * 10 + 1))
        print('there are %d extreme points' % solution_number_selected)
        for k in range(solution_number_selected):
            _model.setParam('SolutionNumber', k)
            v_star = [v[i].Xn for i in range(self.m)]
            u_hat_star = [u_hat[j].Xn for j in range(self.n)]
            extreme_points.append([v_star, u_hat_star])

        return extreme_points

    @staticmethod
    def _Is_Extreme_Point(v, u_hat, graph):
        iss, js = np.where(np.asarray(graph) == 1)
        for i, j in zip(iss, js):
            if v[i] == u_hat[j]:
                pass
            else:
                return False
        iss_0, js_0 = np.where(np.asarray(graph) == 0)
        for i, j in zip(iss_0, js_0):
            if v[i] >= u_hat[j]:
                pass
            else:
                return False
        return True


if __name__ == '__main__':
    from test_example.four_by_eight import *

    print('Sigma - mu*mu \'s eigen values are:', np.linalg.eigvals(sigma_sampled - np.outer(mu_sampled, mu_sampled)))
    cov_real_model = CovRealModel(m, n, f, h, mu_sampled, sigma_sampled, graph, mu_lb, mu_ub, sigma_lb, sigma_ub,
                                  cov_real_param={'bb_params': {'find_init_z': 'v2',
                                                                'select_branching_pos': 'v1'}}, bootstrap=False, )
    cov_real_model.Build_Cov_Real_Model()
    best_model, node_explored = cov_real_model.Solve_Cov_Real_Model()
    I_star = best_model.getVariable('I').level()
    Z_star = best_model.getVariable('Z').level()
    print(I_star)
    print(Z_star)
    print('obj:', best_model.primalObjValue())
