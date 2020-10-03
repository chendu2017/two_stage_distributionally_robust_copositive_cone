from math import sqrt
from typing import Dict
import numpy as np
from co.BB_Node import Node
from mosek.fusion import Model
from co.utils import isAllInteger, ConstructInitZ

TOL = 1e-6


class BranchBound(object):

    def __init__(self, model: Model, mu, sigma, graph, bb_params):
        self.model = model
        self.m, self.n = model.getVariable('I').getShape()[0], model.getVariable('Xi').getShape()[0]
        self.mu, self.sigma = mu, sigma
        self.graph = graph
        self.roads = [(i, j) for i in range(self.m) for j in range(self.n) if graph[i][j] == 1]
        self.bb_params = bb_params
        self.best_node = None
        self.obj_ub = None

    def BB_Solve(self) -> Model:
        # find a feasible solution according to init_z; and set the obj val as an upper bound
        self.best_node = self.__Find_Feasible_Node(self.model.clone())
        self.obj_ub = self.best_node.val

        # construct BB tree root
        root = Node(self.model.clone(), {})

        # deep first search
        self.__Deep_First_Search(root)

    def __Deep_First_Search(self, node: Node):
        node.Update_Z_Constr()
        node.Solve()
        print('current node Z-constr:', node.constr,
              'obj_val:', node.model.primalObjValue(),
              'best_ip_val:', self.obj_ub)

        # if val >= best_IP_val: cut branch
        if node.val < self.obj_ub + TOL:
            if isAllInteger(node.model.getVariable('Z').level()):
                print('Get an integer solution Z:', np.round(node.model.getVariable('Z').level(), 2),
                      f'Update best_ip_val to {node.val}\n')
                self.obj_ub = node.val
                self.best_node = node
            else:
                pos = self.__Select_Branching_Pos(node)
                node.Generate_Child(pos)
                # after generating children, models in the node are useless; Dispose them to save memory
                node.Dispose_Models()
                self.__Deep_First_Search(node.right)
                self.__Deep_First_Search(node.left)

    def __Find_Feasible_Node(self, model: Model):
        init_Z = self.__Find_Init_Z(model)
        init_node = Node(model, init_Z)
        init_node.Update_Z_Constr()
        init_node.Solve()
        print('init node Z-constr:', init_node.constr,
              'obj_val:', init_node.model.primalObjValue(),
              'best_ip_val:', init_node.model.primalObjValue())
        return init_node

    def __Find_Init_Z(self, model: Model) -> Dict[int, int]:
        """
        Find an initial Z solution for co_model.
        v1: randomly select half locations
        v2: select half locations according to average demand upper bound in decreasing order.
        :param model:
        """
        from numpy.random import choice
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

        init_z = ConstructInitZ(self.m, selected)
        return init_z

    def __Select_Branching_Pos(self, node: Node) -> int:
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
            candidates = list(set(range(self.m)) - set(node.constr.keys()))
            pos = choice(candidates)

        if self.bb_params['select_branching_pos'] == 'v2':
            # v2: argmax weighted average alpha_r (r: (i,j)\in roads) for each location i
            sum_alpha_is = []
            alpha = node.model.getVariable('Alpha').level()[:r]
            for i in set(range(self.m))-set(node.constr.keys()):
                sum_alpha_i = sum([alpha[k] if x == i else 0 for k, (x, y) in enumerate(self.roads)])
                sum_alpha_is.append([i, sum_alpha_i])
            pos = sorted(sum_alpha_is, key=lambda x: x[1], reverse=True)[0][0]

        return pos