from typing import Dict

from co.BB_Node import Node
from mosek.fusion import Model
from co.utils import isAllInteger

TOL = 1e-6


class BranchBound(object):

    def __init__(self, model: Model):
        self.model = model
        # find a feasible solution according to init_sol; and set the obj val as an upper bound
        self.best_node = self.__Find_Feasible_Node(self.model.clone())
        self.obj_ub = self.best_node.val

    def BB_Solve(self) -> Model:
        # construct BB tree root
        root = Node(self.model.clone(), {})

        # deep first search
        self.__Deep_First_Search(root)

    def __Deep_First_Search(self, node: Node) -> Node:
        node.Update_Z_Constr()
        node.Solve()
        print('current node Z-constr:', node.constr,
              'obj_val:', node.model.primalObjValue(),
              'best_ip_val:', self.obj_ub)

        # if val >= best_IP_val: cut branch
        if node.val < self.obj_ub + TOL:
            if isAllInteger(node.model.getVariable('Z').level()):
                self.obj_ub = node.val
                self.best_node = node
            else:
                pos = self.__Select_Branching_Pos(node)
                node.Generate_Child(pos)
                # after generating children, models in the node are useless; Dispose them to save memeory
                node.Dispose_Models()
                self.__Deep_First_Search(node.right)
                self.__Deep_First_Search(node.left)

    def __Find_Feasible_Node(self, model: Model):
        init_Z = self.__Find_Init_Z(model)
        init_node = Node(model, init_Z)
        init_node.Update_Z_Constr()
        init_node.Solve()
        return init_node

    @staticmethod
    def __Find_Init_Z(model: Model) -> Dict[int, int]:
        """
        Find an initial Z solution for co_model.
        Temporarily select a half of locations
        :param model:
        """
        from numpy.random import choice
        m = model.getVariable('Z').getShape()[0]
        selected = choice(range(m), size=m // 2, replace=False)
        init_sol = {}
        for i in range(m):
            if i in selected:
                init_sol[i] = 1
            else:
                init_sol[i] = 0
        return init_sol

    @staticmethod
    def __Select_Branching_Pos(node: Node) -> int:
        """
        Ideally, branching position should come from a pricing problem
        temporarily, we randomly select one position from remaining branching position.
        :param node:
        """
        from random import choice
        m = node.model.getVariable('I').getShape()[0]
        candidates = list(set(range(m)) - set(node.constr.keys()))
        pos = choice(candidates)
        return pos