from time import time
from gurobipy.gurobipy import Model, GRB, quicksum
import numpy as np

class SAAModel(object):
    def __init__(self, m, n, f, h, graph, observations, algo_param=None, seed=int(time())):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.graph = graph
        self.d_rs = observations
        self.algo_param = algo_param
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
        self.seed = seed
        self.model = self._Build_Model()

    def _Build_Model(self) -> Model:
        StoModel = Model('StoModel')
        StoModel.setParam('OutputFlag', 0)
        StoModel.modelSense = GRB.MINIMIZE

        m, n = self.m, self.n
        h, f = self.h, self.f
        d_rs = self.d_rs
        d_rs_length = len(d_rs)

        I = StoModel.addVars(m, vtype=GRB.CONTINUOUS, name='I')
        Z = StoModel.addVars(m, vtype=GRB.BINARY, name='Z')
        Transshipment_X = StoModel.addVars(self.roads, d_rs_length, vtype=GRB.CONTINUOUS,
                                           name='Transshipment_X')

        objFunc_holding = quicksum(I[i] * h[i] for i in range(m))
        objFunc_fixed = quicksum(Z[i] * f[i] for i in range(m))
        objFunc_penalty = quicksum(
            d_r[j] - Transshipment_X.sum('*', j, k) for k, d_r in enumerate(d_rs) for j in range(n))
        objFunc = objFunc_holding + objFunc_fixed + (objFunc_penalty / d_rs_length)
        StoModel.setObjective(objFunc)

        # 约束1
        for k in range(d_rs_length):
            StoModel.addConstrs(Transshipment_X.sum(i, '*', k) <= I[i] for i in range(m))

        # 约束2
        for k, d_r in enumerate(d_rs):
            StoModel.addConstrs(Transshipment_X.sum('*', j, k) <= d_r[j] for j in range(n))

        # 约束3 I_i<=M*Z_i
        StoModel.addConstrs(I[i] <= 50000 * Z[i] for i in range(m))

        return StoModel

    def Solve_Model(self) -> Model:
        # 求解评估模型
        self.model.optimize()
        return self.model


if __name__ == '__main__':
    from test_example.four_by_eight import *

    print('Sigma - mu*mu \'s eigen values are:', np.linalg.eigvals(sigma_sampled - np.outer(mu_sampled, mu_sampled)))
    saa_model = SAAModel(m, n, f, h, graph, samples,
                                  algo_param={}
                                  ).Solve_Model()
    I_star = [saa_model.getVarByName(f'I[{i}]').x for i in range(m)],
    Z_star = np.round([saa_model.getVarByName(f'Z[{i}]').x for i in range(m)]).tolist(),
    print(I_star)
    print(Z_star)
    print('obj:', saa_model.ObjVal)
