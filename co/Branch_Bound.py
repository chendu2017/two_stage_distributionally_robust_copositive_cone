from copy import deepcopy
from math import sqrt
from typing import Dict, List
import numpy as np
from gurobipy.gurobipy import GRB, quicksum
from mosek.fusion import Model as mskModel
from mosek.fusion import Domain, Expr
from numerical_study.ns_utils import isAllInteger, Calculate_Sampled_Cov_Matrix
from numpy.random import choice
from mosek.fusion import ProblemStatus
from gurobipy.gurobipy import Model as grbModel

TOL = 1e-5
INF = float('inf')


class BranchBound(object):
    def __init__(self, model: mskModel, mu, sigma, graph, f, h, bb_params):
        self.model = model
        self.pure_model = model.clone()
        self.graph = graph
        self.m, self.n = np.asarray(self.graph).shape
        self.mu, self.sigma = mu, sigma
        self.f, self.h = f, h
        self.roads = [(i, j) for i in range(self.m) for j in range(self.n) if graph[i][j] == 1]
        self.bb_params = bb_params
        self.node_explored = 0
        self.best_model = None
        self.obj_ub = None

    def BB_Solve(self, solve_func=None):
        # find a feasible solution according to init_z; and set the obj val as an upper bound
        model = self._Find_Feasible_Model(solve_func)
        self.best_model = model
        if model.getProblemStatus() == ProblemStatus.PrimalInfeasible \
                or model.getProblemStatus() == ProblemStatus.DualInfeasible:
            self.obj_ub = INF
        else:
            self.obj_ub = self.best_model.primalObjValue()

        # construct BB tree root constr
        root_constr_z = {}

        # deep first search
        self._Deep_First_Search(root_constr_z, solve_func)

    def _Deep_First_Search(self, constr_z: Dict[int, int], solve_func=None):
        pure_model = self.pure_model.clone()
        model = self._Update_Z_Constr(pure_model, constr_z)
        print(constr_z)
        # solve the model
        if solve_func is None:
            model.solve()
            model.getTask().__del__()
        else:
            model = solve_func(model)
        # solving completed

        self.node_explored += 1
        if model.getProblemStatus() == ProblemStatus.PrimalInfeasible \
                or model.getProblemStatus() == ProblemStatus.DualInfeasible:
            model_obj_val = INF
        else:
            model_obj_val = model.primalObjValue()
        # print('current node Z-constr:', constr_z,
        #       'obj_val:', model_obj_val,
        #       'best_ip_val:', self.obj_ub)

        # if val >= best_IP_val: cut branch
        if model_obj_val < self.obj_ub + TOL:
            if isAllInteger(model.getVariable('Z').level()):
                # print('Get an integer solution Z:', np.round(model.getVariable('Z').level()),
                #       f'Update best_ip_val to {model_obj_val}\n')
                self.obj_ub = model_obj_val
                self.best_model.dispose()
                self.best_model = model
            else:
                pos = self._Select_Branching_Pos(model, constr_z)
                model.dispose()  # drop node to release memory
                left = deepcopy(constr_z)
                left[pos] = 0
                right = deepcopy(constr_z)
                right[pos] = 1
                self._Deep_First_Search(right, solve_func=solve_func)
                self._Deep_First_Search(left, solve_func=solve_func)
        else:
            model.dispose()  # cut branch

    def _Find_Feasible_Model(self, solve_func=None):
        pure_model = self.pure_model.clone()
        init_Z = self._Find_Init_Z(pure_model)
        print('init_Z:', init_Z)
        model = self._Update_Z_Constr(pure_model, init_Z)
        if solve_func is None:
            model.solve()
            model.getTask().__del__()
        else:
            model = solve_func(model)
        # print('init node Z-constr:', init_Z,
        #       'obj_val:', model.primalObjValue(),
        #       'best_ip_val:', model.primalObjValue())
        return model

    def _Find_Init_Z(self, model: mskModel) -> Dict[int, int]:
        """
        Find an initial Z solution for co_model.
        v1: randomly select half locations
        v2: select half locations according to average demand upper bound in decreasing order.
        :param model:
        """
        selected = []
        if self.bb_params['find_init_z'] == 'v1':
            # v1 : randomly select half locations
            selected = choice(range(self.m), size=self.m//2, replace=False)

        if self.bb_params['find_init_z'] == 'v2':
            # v2 : choose first half locations : $(\SUM_{j\in\Gamma(i)}\mu_j + \Sigma_{jj})/|\Gamma(i)|$:
            # e.g. decreasingly ranking locations according to average upper bound of all PODs.
            mu, sigma, graph = self.mu, self.sigma, self.graph
            ubs = [mu[j]+sqrt(sigma[j][j]-mu[j]**2) for j in range(self.n)]
            metrics = [[i, np.average(ubs, weights=graph[i])] for i in range(self.m)]
            metrics_ranked = sorted(metrics, key=lambda x: x[1], reverse=True)  # descending
            selected = [pos for pos, _ in metrics_ranked[:self.m//2]]

        if self.bb_params['find_init_z'] == 'v3':
            # use scarf 1958 to determine q_{ij}^*, and then find the most cheap storage plan by solving MIP
            mu, std = self.mu, np.asarray([sqrt(self.sigma[j][j]-self.mu[j]**2) for j in range(self.n)])
            q = np.zeros((self.m, self.n))
            for (i, j) in self.roads:
                constant = sqrt((1-self.h[i])/self.h[i])
                if mu[j]/std[j] >= 1/constant:
                    q[i, j] = max(0, mu[j] + std[j]/2*(constant - 1/constant))
            # MIP model
            warm_up_model = grbModel('warm_up_model')
            warm_up_model.setParam('OutputFlag', 0)
            warm_up_model.modelSense = GRB.MINIMIZE
            Z = warm_up_model.addVars(self.m, vtype=GRB.BINARY, name='Z')
            X = warm_up_model.addVars(self.roads, vtype=GRB.BINARY, name='x')
            obj = quicksum(Z[i]*self.f[i] for i in range(self.m)) + quicksum(q[i, j]*self.h[i]*X[i, j]
                                                                             for (i, j) in self.roads)
            warm_up_model.setObjective(obj)
            warm_up_model.addConstrs(X.sum('*', j) >= 1 for j in range(self.n))
            warm_up_model.addConstrs(X[i, j] <= Z[i] for (i, j) in self.roads)
            warm_up_model.optimize()
            selected = [i for i in range(self.m) if Z[i].x == 1]

        init_z = self._ConstructInitZ(self.m, selected)
        return init_z

    def _Select_Branching_Pos(self, solved_model: mskModel, constr_z: Dict[int, int]) -> int:
        """
        Ideally, branching position should come from a pricing problem
        v1: randomly select one position from remaining branching position.
        v2: argmax weighted average alpha_r (r: (i,j)\in roads) for each location i
        :param node:
        """
        from random import choice
        r = len(self.roads)

        if self.bb_params['select_branching_pos'] == 'v1':
            # v1: randomly select one position from remaining branching position.
            candidates = list(set(range(self.m)) - set(constr_z.keys()))
            pos = choice(candidates)

        if self.bb_params['select_branching_pos'] == 'v2':
            # v2: argmax weighted average alpha_r (r: (i,j)\in roads) for each location i
            sum_alpha_is = []
            alpha = solved_model.getVariable('Alpha').level()[:r]
            for i in set(range(self.m))-set(constr_z.keys()):
                sum_alpha_i = sum([alpha[k] if x == i else 0 for k, (x, y) in enumerate(self.roads)])
                sum_alpha_is.append([i, sum_alpha_i])
            pos = sorted(sum_alpha_is, key=lambda x: x[1], reverse=True)[0][0]

        return pos

    @staticmethod
    def _Update_Z_Constr(pure_model: mskModel, constr_z: Dict[int, int]) -> mskModel:
        Z = pure_model.getVariable('Z')
        if len(constr_z) == 1:
            for key, value in constr_z.items():  # only one iteration
                pure_model.constraint('BB',  Z.index(key), Domain.equalsTo(value))
        if len(constr_z) >= 2:
            expression = Expr.vstack([Z.index(key) for key in constr_z.keys()])
            values = [value for key, value in constr_z.items()]
            pure_model.constraint('BB', expression, Domain.equalsTo(values))
        return pure_model

    @staticmethod
    def _ConstructInitZ(m: int, locations: List[int]) -> Dict[int, int]:
        init_z = {}
        for i in range(m):
            if i in locations:
                init_z[i] = 1
            else:
                init_z[i] = 0
        return init_z

