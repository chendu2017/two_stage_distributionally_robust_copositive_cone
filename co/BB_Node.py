from typing import Dict
from mosek.fusion import Domain, SolutionStatus, Model, Expr


class Node(object):
    sol: Dict[str, list]
    val: float

    def __init__(self, model: Model, constr: Dict[int, int]):
        self.model = model
        self.pure_model = model.clone()
        self.left = None
        self.right = None
        self.val = float('inf')
        self.sol = None
        self.constr = constr

    def Remove_Z_Constr(self):
        constr = self.model.getConstraint('BB')
        if constr is not None:
            constr.remove()

    def Update_Z_Constr(self):
        self.Remove_Z_Constr()
        Z = self.model.getVariable('Z')
        if len(self.constr) == 1:
            for key, value in self.constr.items():  # only one iteration
                self.model.constraint('BB',  Z.index(key), Domain.equalsTo(value))
        if len(self.constr) >= 2:
            expression = Expr.vstack([Z.index(key) for key in self.constr.keys()])
            values = [value for key, value in self.constr.items()]
            self.model.constraint('BB', expression, Domain.equalsTo(values))

    def Solve(self):
        self.model.solve()
        if self.model.getPrimalSolutionStatus() == SolutionStatus.Optimal:
            self.Record_Sol()

    def Dispose_Models(self):
        self.model.dispose()
        self.pure_model.dispose()

    def Record_Sol(self):
        self.sol = {'I': self.model.getVariable('I').level().tolist(),
                    'Z': self.model.getVariable('Z').level().tolist()}
        self.val = self.model.primalObjValue()

    def Generate_Child(self, pos):
        from copy import deepcopy
        lc_constr = deepcopy(self.constr)
        lc_constr[pos] = 0
        rc_constr = deepcopy(self.constr)
        rc_constr[pos] = 1
        self.left = Node(self.pure_model.clone(), lc_constr)
        self.right = Node(self.pure_model.clone(), rc_constr)



