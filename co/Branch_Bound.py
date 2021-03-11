from copy import deepcopy
from math import sqrt
from typing import Dict, List
import numpy as np
from mosek.fusion import Model, Domain, Expr
from numerical_study.ns_utils import isAllInteger
from numpy.random import choice

TOL = 1e-6


class BranchBound(object):
    def __init__(self, model: Model, mu, sigma, graph, bb_params):
        self.model = model
        self.pure_model = model.clone()
        self.m, self.n = model.getVariable('I').getShape()[0], model.getVariable('Xi').getShape()[0]
        self.mu, self.sigma = mu, sigma
        self.graph = graph
        self.roads = [(i, j) for i in range(self.m) for j in range(self.n) if graph[i][j] == 1]
        self.bb_params = bb_params
        self.node_explored = 0
        self.best_model = None
        self.obj_ub = None

    def BB_Solve(self) -> Model:
        # find a feasible solution according to init_z; and set the obj val as an upper bound
        self.best_model = self.__Find_Feasible_Model()
        self.obj_ub = self.best_model.primalObjValue()

        # construct BB tree root constr
        root_constr_z = {}

        # deep first search
        self.__Deep_First_Search(root_constr_z)

    def __Deep_First_Search(self, constr_z: Dict[int, int]):
        pure_model = self.pure_model.clone()
        model = self.__Update_Z_Constr(pure_model, constr_z)
        model.solve()
        model.getTask().__del__()  # delete the underlying optimization task, which contributes to memory leakage
        self.node_explored += 1
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
                pos = self.__Select_Branching_Pos(model, constr_z)
                model.dispose()  # drop node to release memory
                left = deepcopy(constr_z)
                left[pos] = 0
                right = deepcopy(constr_z)
                right[pos] = 1
                self.__Deep_First_Search(right)
                self.__Deep_First_Search(left)
        else:
            model.dispose()  # cut branch

    def __Find_Feasible_Model(self):
        pure_model = self.pure_model.clone()
        init_Z = self.__Find_Init_Z(pure_model)
        model = self.__Update_Z_Constr(pure_model, init_Z)
        model.solve()
        model.getTask().__del__()
        # print('init node Z-constr:', init_Z,
        #       'obj_val:', model.primalObjValue(),
        #       'best_ip_val:', model.primalObjValue())
        return model

    def __Find_Init_Z(self, model: Model) -> Dict[int, int]:
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

        init_z = self.__ConstructInitZ(self.m, selected)
        return init_z

    def __Select_Branching_Pos(self, solved_model: Model, constr_z: Dict[int, int]) -> int:
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
    def __Update_Z_Constr(pure_model: Model, constr_z: Dict[int, int]) -> Model:
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
    def __ConstructInitZ(m: int, locations: List[int]) -> Dict[int, int]:
        init_z = {}
        for i in range(m):
            if i in locations:
                init_z[i] = 1
            else:
                init_z[i] = 0
        return init_z

