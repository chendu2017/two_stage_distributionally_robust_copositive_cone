from gurobipy.gurobipy import Model, GRB, quicksum
from numpy import matmul
from time import time


class LDRModel(object):
    model: Model

    def __init__(self, m, n, f, h, mu, sigma, graph, ldr_params=None, seed=int(time())):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.mu, self.sigma = mu, sigma
        self.graph = graph
        self.ldr_params = ldr_params
        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
        self.seed = seed
        self.model = None

    def Build_LDR_Model(self):
        """
        LDR for location-inventpry problem is only feasible when all affine coefficients equal to 0, which violates from
        what we want.
        For more info about how to design a reasonable prepositioning network,
        see: http://web.hec.ca/pages/erick.delage/LRC_TRO.pdf

        The above paper assume support set is a convex polyhedron characterized by linear constraints.
        """
        m, n = self.m, self.n
        f, h = self.f, self.h
        mu, sigma = self.mu, self.sigma
        roads = self.roads
        INF = float('inf')
        f_ks, K = self.ldr_params['f_k'], len(self.ldr_params['f_k'])

        # declare model
        ldr_model = Model()

        # decision variables
        I = ldr_model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name='I')
        Z = ldr_model.addVars(m, vtype=GRB.BINARY, lb=0.0, name='Z')
        t = ldr_model.addVars(n, vtype=GRB.CONTINUOUS, lb=-INF, name='t')
        gamma = ldr_model.addVar(lb=-INF, vtype=GRB.CONTINUOUS, name='gamma')
        s = ldr_model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name='s')
        phi = ldr_model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name='phi')
        psi = ldr_model.addVars(K, vtype=GRB.CONTINUOUS, lb=-INF, name='psi')
        xi = ldr_model.addVars(K, vtype=GRB.CONTINUOUS, lb=-INF, name='xi')
        eta = ldr_model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=0.0, name='eta')
        tau = ldr_model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=-INF, name='tau')
        theta = ldr_model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=-INF, name='theta')
        w = ldr_model.addVars(m, K, vtype=GRB.CONTINUOUS, lb=0.0, name='w')
        lambd = ldr_model.addVars(m, K, vtype=GRB.CONTINUOUS, lb=-INF, name='lambda')
        delta = ldr_model.addVars(m, K, vtype=GRB.CONTINUOUS, lb=-INF, name='delta')
        alpha_0 = ldr_model.addVars(roads, vtype=GRB.CONTINUOUS, lb=-INF, name='alpha_0')
        alpha = ldr_model.addVars(roads, n, vtype=GRB.CONTINUOUS, lb=-INF, name='alpha')
        beta = ldr_model.addVars(roads, K, vtype=GRB.CONTINUOUS, lb=-INF, name='beta')

        # objective function
        obj1 = quicksum([f[i]*Z[i] for i in range(m)])
        obj2 = quicksum([h[i]*I[i] for i in range(m)])
        obj3 = quicksum([mu[i]*t[i] for i in range(m)])
        obj4 = quicksum(s[k]*matmul(matmul(f_k, sigma), f_k) for k, f_k in enumerate(f_ks))
        obj5 = gamma
        ldr_model.setObjective(obj1 + obj2 + obj3 + obj4 + obj5)
        ldr_model.modelSense = GRB.MINIMIZE

        # constraint
        # c1
        ldr_model.addConstr(gamma + alpha_0.sum() >= 1/2*(phi.sum() - psi.sum()),
                            name='c1')

        # c2
        ldr_model.addConstrs((1/2*(phi[k]+psi[k]) == beta.sum('*', '*', k) + s[k]*len(roads) for k in range(K)),
                             name='c2')

        # c3
        a_star_star = [alpha.sum('*', '*', l) for l in range(n)]
        ldr_model.addConstrs((xi[k]*f_ks[k][l] <= t[l]+a_star_star[l]-1 for k in range(K) for l in range(n)),
                             name='c3')

        # c4
        ldr_model.addConstrs((-alpha_0.sum('*', j) >= 1/2*(eta.sum(j, '*')-tau.sum(j, '*')) for j in range(n)),
                             name='c4')

        # c5
        ldr_model.addConstrs((-1/2*(eta[j, k]+tau[j, k]) == beta.sum('*', j, k) for k in range(K) for j in range(n)),
                             name='c5')

        # c6
        for j in range(n):
            a_star_j = [alpha.sum('*', j, l) if l != j else alpha.sum('*', j, l)-1 for l in range(n)]
            for k, f_k in enumerate(f_ks):
                ldr_model.addConstrs(-theta[j, k]*f_k[l] >= a_star_j[l] for l in range(n))

        # c7
        ldr_model.addConstrs((I[i]-alpha_0.sum(i, '*') >= 1/2*(w.sum(i, '*')-lambd.sum(i, '*')) for i in range(m)),
                             name='c7')

        # c8
        ldr_model.addConstrs((-1/2*(w[i, k]+lambd[i, k]) == beta.sum(i, '*', k) for i in range(m) for k in range(K)),
                             name='c8')

        # c9
        for i in range(m):
            a_i_star = [alpha.sum(i, '*', l) for l in range(n)]
            for k, f_k in enumerate(f_ks):
                ldr_model.addConstrs(-delta[i, k]*f_k[l] >= a_i_star[l] for l in range(n))

        # c10 I,Z
        ldr_model.addConstrs(I[i] <= 10000*Z[i] for i in range(m))

        # c11
        ldr_model.addConstrs(phi[k]*phi[k] >= psi[k]*psi[k] + xi[k]*xi[k] for k in range(K))
        ldr_model.addConstrs(eta[i, k]*eta[i, k] >= tau[i, k]*tau[i, k] + theta[i, k]*theta[i, k]
                             for k in range(K) for i in range(m))
        ldr_model.addConstrs(w[i, k]*w[i, k] >= lambd[i, k]*lambd[i, k] + delta[i, k]*delta[i, k]
                             for k in range(K) for i in range(m))

        # update
        ldr_model.update()
        ldr_model.optimize()
        # print(I)
        # print(Z)
        # print(alpha_0)
        # print(alpha)
        # print(beta)
        # print(ldr_model.ObjVal)
        self.model = ldr_model

    def Solve_LDR_Model(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        return self.model


if __name__ == '__main__':
    ldr = LDRModel(m, n, f, h, first_moment, second_moment, graph, ldr_params={'f_k': [[1,0,0,0],
                                                                                       [0,1,0,0],
                                                                                       [0,0,1,0],
                                                                                       [0,0,0,1],
                                                                                       [1,1,1,1],
                                                                                       [1,0,1,0],
                                                                                       [0,1,0,1],
                                                                                       [0,1,1,0],
                                                                                       [1,0,0,1]]})
    ldr.Build_LDR_Model()
    ldr = ldr.Solve_LDR_Model()
    print(ldr.ObjVal)
    pass

