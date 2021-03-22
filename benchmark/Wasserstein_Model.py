from mosek.fusion import Model, Domain, Expr, ObjectiveSense
from itertools import product
import numpy as np
from gurobipy.gurobipy import Model as grbModel, GurobiError, GRB
from gurobipy.gurobipy import GRB, quicksum
from time import time
from benchmark.Cov_Real_Model import CovRealModel
from benchmark.SAA_Model import SAAModel
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


class WassersteinModel(SAAModel):
    def __init__(self, m, n, f, h, graph, observations, algo_param=None, seed=int(time()), z_constr=None):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.graph = graph
        self.d_rs = observations
        self.d_bar = np.amax(observations, axis=0) * 2
        self.algo_param = algo_param
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
        self.seed = seed
        self.model = self._Build_Model(z_constr)

    def Solve_Model(self) -> grbModel:
        self.model.optimize()
        return self.model

    @deprecated(reason='see Hanasusanto G A, Kuhn D. Conic programming reformulations of two-stage distributionally '
                       'robust linear programs over wasserstein balls[J]. Operations Research, 2018, 66(3): 849-869.'
                       'to learn how to use Wasserstein Ball ambiguity set')
    def _Build_Model(self, z_constr=None) -> grbModel:
        extreme_points = self._Find_Extreme_Points_by_Solving_MIP(z_constr)
        K = len(extreme_points)
        WassModel = grbModel('WassModel')
        WassModel.setParam('OutputFlag', 0)
        WassModel.modelSense = GRB.MINIMIZE

        m, n = self.m, self.n
        h, f = self.h, self.f
        d_rs = self.d_rs
        d_rs_length = len(d_rs)

        I = WassModel.addVars(m, vtype=GRB.CONTINUOUS, lb=0, name='I')
        Z = WassModel.addVars(m, vtype=GRB.BINARY, name='Z')
        Lambda = WassModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name='Lambda')
        s = WassModel.addVars(d_rs_length, vtype=GRB.CONTINUOUS, lb=-INF, name='s')
        z = WassModel.addVars(d_rs_length, K, n, vtype=GRB.CONTINUOUS, lb=-INF, name='z')
        gamma = WassModel.addVars(d_rs_length, K, n, vtype=GRB.CONTINUOUS, lb=0, name='gamma')
        if self.algo_param['wasserstein_p'] == 'inf':
            z_abs = WassModel.addVars(d_rs_length, K, n, vtype=GRB.CONTINUOUS, lb=0, name='z_abs')

        # objective
        objFunc_holding = quicksum(I[i] * h[i] for i in range(m))
        objFunc_fixed = quicksum(Z[i] * f[i] for i in range(m))
        objFunc_penalty_1 = Lambda * self.algo_param['wasserstein_ball_radius']
        objFunc_penalty_2 = s.sum() / d_rs_length
        objFunc = objFunc_holding + objFunc_fixed + objFunc_penalty_1 + objFunc_penalty_2
        WassModel.setObjective(objFunc)

        # 约束1 - wasserstein ball
        for i in range(d_rs_length):
            for k, (v, u_hat) in enumerate(extreme_points):
                WassModel.addConstr(
                    quicksum(gamma[i, k, j] * self.d_bar[j] for j in range(n))
                    + quicksum(z[i, k, j] * d_rs[i, j] for j in range(n))
                    - quicksum(I[i] * v[i] for i in range(m))
                    <= s[i])
                WassModel.addConstrs(gamma[i, k, j] >= u_hat[j] - z[i, k, j] for j in range(n))

                # dual_norm(z[i, :], p) <= Lambda
                if self.algo_param['wasserstein_p'] == 1:
                    WassModel.addConstrs(z[i, k, j] <= Lambda for j in range(n))
                if self.algo_param['wasserstein_p'] == 2:
                    WassModel.addConstr(quicksum(z[(i, k, j)] ** 2 for j in range(n)) <= Lambda * Lambda)
                if self.algo_param['wasserstein_p'] == 'inf':
                    WassModel.addConstr(z_abs.sum(i, '*') <= Lambda)
                    WassModel.addConstrs(z_abs[i, k, j] >= z[i, k, j] for j in range(n))
                    WassModel.addConstrs(z_abs[i, k, j] >= -z[i, k, j] for j in range(n))

        # 约束2 I_i<=M*Z_i
        WassModel.addConstrs(I[i] <= 20000 * Z[i] for i in range(m))
        if z_constr is None:
            pass
        else:
            WassModel.addConstrs(Z[i] == z_constr[i] for i in range(m))

        return WassModel

    def _Find_Extreme_Points_by_Solving_MIP(self, z=None):
        if z is None:
            z = [1] * self.m

        extreme_points = []

        _model = grbModel('subModel')
        _model.setParam('OutputFlag', 1)
        _model.modelSense = GRB.MAXIMIZE

        # Variables
        v = _model.addVars(self.m, vtype=GRB.BINARY, obj=0, name='v')
        u_hat = _model.addVars(self.n, vtype=GRB.BINARY, obj=0, name='u_hat')

        # Constraints
        roads_z = [(i, j) for (i, j) in self.roads if z[i] == 1]
        for (i, j) in roads_z:
            _model.addConstr(v[i] >= u_hat[j])

        # Solve
        _model.setParam('OutputFlag', 0)
        _model.setParam('MIPFocus', 1)  # focus on finding feasible solutions
        _model.setParam('PoolSearchMode', 2)  # search for more solutions, but in a non-systematic way
        _model.setParam('PoolSolutions', self.algo_param['max_num_extreme_points'])
        _model.optimize()

        # get some feasible solutions
        solution_number = _model.SolCount
        solution_number_selected = int(min(solution_number, self.algo_param['max_num_extreme_points']))
        print('there are %d extreme points' % solution_number_selected)
        for k in range(solution_number_selected):
            _model.setParam('SolutionNumber', k)
            v_star = [v[i].Xn for i in range(self.m)]
            u_hat_star = [u_hat[j].Xn for j in range(self.n)]
            extreme_points.append([v_star, u_hat_star])

        return extreme_points


if __name__ == '__main__':
    from test_example.four_by_eight import *

    print('Sigma - mu*mu \'s eigen values are:', np.linalg.eigvals(sigma_sampled - np.outer(mu_sampled, mu_sampled)))
    results = {}
    # for k, z_constr in enumerate(product([0, 1], repeat=m)):
    wass_model = WassersteinModel(m, n, f, h, graph, samples,
                                  algo_param={'wasserstein_ball_radius': 10,
                                              'wasserstein_p': 1,
                                              'max_num_extreme_points': 10000}
                                  ).Solve_Model()

    I_star = [wass_model.getVarByName(f'I[{i}]').x for i in range(m)]
    Z_star = np.round([wass_model.getVarByName(f'Z[{i}]').x for i in range(m)]).tolist()
    Lambda_star = wass_model.getVarByName('Lambda').x
    s_star = [wass_model.getVarByName(f's[{i}]').x for i in range(m)]
    print(I_star)
    print(Z_star)
    print(Lambda_star)
    print(s_star)
    # results[tuple(z_constr)] = wass_model.ObjVal
    results[0] = {'I': I_star,
                  'Z': Z_star,
                  "lambda": Lambda_star,
                  's': s_star}
    print('obj:', wass_model.ObjVal)
