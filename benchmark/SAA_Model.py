from gurobipy.gurobipy import Model, GRB, quicksum


class SAAModel(object):
    def __init__(self, m, n, f, h, d_rs, graph, saa_params=None):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.graph = graph
        self.d_rs = d_rs
        self.saa_params = saa_params
        self.model = None
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]

    def SolveStoModel(self) -> Model:
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
        objFunc_penalty = quicksum(d_r[j] - Transshipment_X.sum('*', j, k) for k, d_r in d_rs.items() for j in range(n))
        objFunc = objFunc_holding + objFunc_fixed + (objFunc_penalty/d_rs_length)
        StoModel.setObjective(objFunc)

        # 约束1
        for k in range(d_rs_length):
            StoModel.addConstrs(Transshipment_X.sum(i, '*', k) <= I[i] for i in range(m))

        # 约束2
        for k, d_r in d_rs.items():
            StoModel.addConstrs(Transshipment_X.sum('*', j, k) <= d_r[j] for j in range(n))

        # 约束3 I_i<=M*Z_i
        StoModel.addConstrs(I[i] <= 20000 * Z[i] for i in range(m))

        # 求解评估模型
        StoModel.optimize()
        self.model = StoModel
        return StoModel


if __name__ == '__main__':
    from test_example.four_by_four_d_rs import m, n, f, h, d_rs, graph
    saa_model = SAAModel(m, n, f, h, d_rs, graph).SolveStoModel()
    print([saa_model.getVarByName(f'I[{i}]').x for i in range(m)])
