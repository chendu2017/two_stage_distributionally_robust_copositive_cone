from gurobipy.gurobipy import Model, GRB, quicksum
from time import time

class DETModel(object):
    def __init__(self, m, n, f, h, graph, det_param=None, seed=int(time())):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.graph = graph
        self.det_param = det_param
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
        self.seed = seed
        self.model = None

    def SolveDetModel(self) -> Model:
        DetModel = Model('StoModel')
        DetModel.setParam('OutputFlag', 0)
        DetModel.modelSense = GRB.MINIMIZE

        m, n = self.m, self.n
        h, f = self.h, self.f
        mu = self.det_param['mu']

        I = DetModel.addVars(m, vtype=GRB.CONTINUOUS, name='I')
        Z = DetModel.addVars(m, vtype=GRB.BINARY, name='Z')
        Transshipment_X = DetModel.addVars(self.roads, vtype=GRB.CONTINUOUS, name='Transshipment_X')

        objFunc_holding = quicksum(I[i] * h[i] for i in range(m))
        objFunc_fixed = quicksum(Z[i] * f[i] for i in range(m))
        objFunc_penalty = quicksum(mu[j] - Transshipment_X.sum('*', j) for j in range(n))
        objFunc = objFunc_holding + objFunc_fixed + objFunc_penalty
        DetModel.setObjective(objFunc)

        # 约束1
        DetModel.addConstrs(Transshipment_X.sum(i, '*') <= I[i] for i in range(m))

        # 约束2
        DetModel.addConstrs(Transshipment_X.sum('*', j) <= mu[j] for j in range(n))

        # 约束3 I_i<=M*Z_i
        DetModel.addConstrs(I[i] <= 20000 * Z[i] for i in range(m))

        # 求解评估模型
        DetModel.optimize()
        self.model = DetModel
        return DetModel


if __name__ == '__main__':
    from test_example.four_by_four_d_rs import m, n, f, h, graph, saa_param
    det_model = DETModel(m, n, f, h, graph, {'mu': [20, 20, 20, 20]}).SolveDetModel()
    print([det_model.getVarByName(f'I[{i}]').x for i in range(m)])