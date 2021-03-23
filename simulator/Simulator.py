from typing import Dict, List, Any
from gurobipy.gurobipy import tuplelist, Model, GRB
from numerical_study.ns_utils import isAllInteger
import numpy as np


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

    def setDemand_Realizations(self, d_rs: np.ndarray):
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

        for k, d_r in enumerate(d_rs):
            result = self._Simulate(d_r.tolist())
            self.results[k] = result
        return self.results

    def _Simulate(self, d_r):
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
        x = np.zeros((self.m, self.n))
        for (i, j) in self.roads_Z:
            x[i, j] = X[(i, j)].X
        return {'d_r': d_r, 'X': x.tolist()}


if __name__ == '__main__':
    s = Simulator(2, 2, [[1, 0],[1, 1]])
    s.setSol({'I': [10, 30],
              'Z': [1, 1]})
    s.Run_Simulations(d_rs={1: [15, 20]})

