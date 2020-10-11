from typing import Dict, List, Any
from gurobipy.gurobipy import tuplelist, Model, GRB
from utils import isAllInteger


class Simulator(object):
    def __init__(self, m, n, graph):
        self.m, self.n = m, n
        self.graph = graph
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
        self.roads_Z = None
        self.d_rs = None
        self.I = None
        self.Z = None
        self.results = None

    def setDemand_Realizations(self, d_rs: Dict[int, List[float]]):
        self.d_rs = d_rs

    def setSol(self, sol: Dict[str, List[float]]):
        assert isAllInteger(sol['Z']), 'Zs are not all integers'
        self.I = sol['I']
        self.Z = [round(z) for z in sol['Z']]
        self.roads_Z = [(i, j) for (i, j) in self.roads if self.Z[i] == 1]

    def Run_Simulations(self, d_rs=None) -> Dict[int, Dict[str, Any]]:
        self.results = {}
        if d_rs is None:
            d_rs = self.d_rs

        for k, d_r in d_rs.items():
            result = self.__Simulate(d_r)
            self.results[k] = result
        return self.results

    def __Simulate(self, d_r):
        # define the model
        model = Model('evaluation_problem')
        model.setParam('OutputFlag', 0)
        model.modelSense = GRB.MINIMIZE

        # decision variables
        X = model.addVars(tuplelist(self.roads_Z), vtype=GRB.CONTINUOUS, lb=0.0, name='X')

        # objective function
        objFunc_penal = sum(d_r) - X.sum()
        model.setObjective(objFunc_penal)

        # constraints
        model.addConstrs(X.sum('*', j) <= d_r[j] for j in range(self.n))
        model.addConstrs(X.sum(i, '*') <= self.I[i] for i in range(self.m))

        # solve
        model.optimize()

        # record
        return {'d_r': d_r, 'X': {f'{r}': X[r].X for r in self.roads_Z}}


if __name__ == '__main__':
    s = Simulator(2, 2, [[1, 0],[1, 1]])
    s.setSol({'I': [10, 30],
              'Z': [1, 1]})
    s.Run_Simulations(d_rs={1: [15, 20]})

