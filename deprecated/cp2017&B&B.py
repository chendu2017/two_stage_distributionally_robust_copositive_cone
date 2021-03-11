# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:28:03 2019

@author: chend
"""

import numpy as np
import copy

fixed_cost = [10, 11, 12, 13, 14, 15, 16, 17]
holding_cost = [0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]

MU_d = np.asarray([50, 30, 50, 40, 60, 50, 55, 65])
SIGMA_d = np.array([15, 10, 12, 13, 17, 14, 16, 11])

M = len(fixed_cost)
N = len(MU_d)

D = np.random.normal(loc=MU_d, scale=SIGMA_d, size=[1000000, N])
D[D <= 0] = 0
D_Mat = np.matrix(D)
second_moment = (D_Mat.T * D_Mat) / 1000000

SUPPORT_d = np.array([[0, 100],
                      [0, 100],
                      [0, 100],
                      [0, 100],
                      [0, 100],
                      [0, 100],
                      [0, 100],
                      [0, 100]])

ACCESSGRAPH = np.array([[1, 1, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0, 0, 1, 1],
                        [1, 1, 0, 1, 0, 1, 1, 0],
                        [1, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 1, 1]
                        ])
ROAD = [(row + 1, col + 1) for row, col in zip(np.where(ACCESSGRAPH == 1)[0], np.where(ACCESSGRAPH == 1)[1])]

'''
p = [\hat{u} v s u^\dagger v^\dagger s^\dagger]
A = [a_1, a_2, ... , a_{r+M+N+r}]
'''

# --- a_r
r = len(ROAD)
A = np.zeros(shape=(r + N + M + r + 1, 2 * (N + M + r) + 1))  # 多一行一列，方便输入矩阵，后面删掉
for row in range(1, 1 + r + N + M + r):

    if row <= r:
        i, j = ROAD[row - 1]

        A[row, j] = 1
        A[row, N + i] = -1
        A[row, N + M + row] = 1

    if r < row <= r + N:
        j = row - r

        A[row, j] = 1
        A[row, j + N + M + r] = 1

    if r + N < row <= r + N + M:
        i = row - r - N

        A[row, N + i] = 1
        A[row, N + i + M + N + r] = 1

    if r + N + M < row <= r + N + M + r:
        tiao = row - r - N - M

        A[row, N + M + tiao] = 1
        A[row, N + M + r + N + M + tiao] = 1

A = A[1:, 1:].transpose()

b = np.asarray([0] * r + [1] * (N + M + r))

import mosek
from mosek.fusion import *

A_Mat = Matrix.dense(A)


def GetSolvedCOModel(Z_Info=False):
    COModel = Model()

    # --- Decision Variable
    # M1 matrix
    Tau = COModel.variable('Tau', 1, Domain.unbounded())  # scalar
    Zeta = COModel.variable('Zeta', N, Domain.unbounded())  # n by 1 vector
    Theta = COModel.variable('Theta', 2 * (M + N + r), Domain.unbounded())  # 2*(M+N+r) by 1 vector
    Alpha = COModel.variable('Alpha', r + N + M + r, Domain.unbounded())  # r+N+M+r by 1 vector
    Beta = COModel.variable('Beta', r + N + M + r, Domain.unbounded())  # r+N+M+r by 1 vector
    Eta = COModel.variable('Eta', [N, N], Domain.unbounded())  # N by N matrix
    Phi = COModel.variable('Phi', 2 * (M + N + r), Domain.unbounded())  # 2*(M+N+r) by 1 vector
    W = COModel.variable('W', [2 * (M + N + r), 2 * (M + N + r)], Domain.unbounded())  # 2*(M+N+r) by 2*(M+N+r) matrix
    Psi = COModel.variable('Psi', [2 * (M + N + r), N], Domain.unbounded())

    Z = COModel.variable('Z', M, Domain.inRange(0.0, 1.0))
    I = COModel.variable('I', M, Domain.greaterThan(0.0))

    # M2 matrix
    M_2 = COModel.variable([1 + N + 2 * (N + M + r), 1 + N + 2 * (N + M + r)],
                           Domain.greaterThan([[0.0] * (2 * (M + N + r) + 1 + N)
                                               ] * (2 * (M + N + r) + 1 + N)))

    # --- Objective Fcuntion
    obj_1 = Expr.dot(b.tolist(), Alpha)
    obj_2 = Expr.dot(b.tolist(), Beta)
    obj_3 = Expr.dot([1], Expr.add(Tau, M_2.index([0, 0])))
    obj_4 = Expr.add(Expr.dot(MU_d.tolist(), Expr.add(Zeta, Expr.transpose(M_2.slice([0, 1], [1, 1 + N])))),
                     Expr.dot(MU_d.tolist(), Expr.add(Zeta, M_2.slice([1, 0], [1 + N, 1]))))
    obj_5 = Expr.dot(second_moment.tolist(), Expr.add(Eta, M_2.slice([1, 1], [1 + N, 1 + N])))

    obj_6 = Expr.dot(fixed_cost, Z)
    obj_7 = Expr.dot(holding_cost, I)
    COModel.objective(ObjectiveSense.Minimize, Expr.add([obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7]))

    # --- Constraints

    # all expressions represented by serveral variables must be greater than or equal to 0
    # 1. 
    _expr = Expr.sub(Expr.sub(Expr.mul(A_Mat, Alpha), Theta), Expr.mul(2, Phi));
    _expr = Expr.reshape(_expr, 2 * (M + N + r), 1)
    _expr = Expr.sub(_expr, M_2.slice([1 + N, 0], [1 + N + 2 * (M + N + r), 1]))
    _expr = Expr.sub(_expr, Expr.transpose(M_2.slice([0, 1 + N], [1, 1 + N + 2 * (M + N + r)])))
    _expr = Expr.reshape(_expr, 2 * (M + N + r))
    _expr_rhs = Expr.vstack(Expr.constTerm([0] * N), Expr.mul(-1, I), Expr.constTerm([0] * (r + M + N + r)))
    COModel.constraint(Expr.sub(_expr, _expr_rhs), Domain.equalsTo(0))
    del _expr

    # 2. 
    _first_term = Expr.add([Expr.mul(Beta.index(col), np.outer(A[:, col], A[:, col]).tolist())
                            for col in range(r + N + M + r)])
    _second_term = Matrix.sparse(2 * (M + N + r), 2 * (M + N + r))
    for k in range(2 * (M + N + r)):
        _second_term = Expr.add(_second_term,
                                Expr.mul(Theta.index(k), Matrix.sparse(2 * (M + N + r), 2 * (M + N + r), [k], [k], [1]))
                                )
    _expr = Expr.sub(Expr.add(_first_term, _second_term),
                     Expr.add(W, M_2.slice([1 + N, 1 + N], [1 + N + 2 * (M + N + r), 1 + N + 2 * (M + N + r)])))
    COModel.constraint(_expr, Domain.equalsTo([[0] * 2 * (M + N + r)] * (2 * (M + N + r))))
    del _expr

    # 3.
    _expr_rhs = np.zeros(shape=[2 * (M + N + r), N])
    for k in range(N):
        _expr_rhs[k, k] = 1
    _expr = Expr.sub(Expr.mul(-2, Psi), M_2.slice([1 + N, 1 + N], [1 + N + 2 * (M + N + r), 1 + N + N]))
    _expr = Expr.sub(_expr, Expr.transpose(M_2.slice([1 + N, 1 + N], [1 + N + N, 1 + N + 2 * (M + N + r)])))
    COModel.constraint(_expr, Domain.equalsTo(_expr_rhs.tolist()))
    del _expr

    # 4. I constraints
    COModel.constraint(Expr.sub(Expr.mul(500, Z), I), Domain.greaterThan([0] * M))

    # M1 matrix is PSD
    COModel.constraint(Expr.vstack(
        Expr.hstack(Tau.transpose(), Zeta.transpose(), Phi.transpose()),
        Expr.hstack(Expr.reshape(Zeta, [N, 1]), Eta, Psi.transpose()),
        Expr.hstack(Phi, Psi, W),
    ), Domain.inPSDCone())

    # add Z_Info related constraints
    if Z_Info:
        if len(Z_Info) == 1:
            location = list(Z_Info.keys())[0]
            NEW_Z = Z.index(location - 1)
            b_tmp = float(Z_Info[location])
        else:
            NEW_Z = Expr.vstack([Z.index(i - 1) for i in Z_Info.keys()])
            b_tmp = [float(Z_Info[i]) for i in Z_Info.keys()]
        new_constr = COModel.constraint(NEW_Z, Domain.equalsTo(b_tmp))

    # solve model
    COModel.solve()

    # remove Z_Info related constraint
    if Z_Info:
        new_constr.update(Expr.mul(0.0, NEW_Z))
        if len(Z_Info) == 1:
            new_constr.update([b_tmp])
        else:
            new_constr.update(b_tmp)

    return COModel


class BranchBoundNode():

    def __init__(self, BaseModel, Z_Info):
        self.CurrentCOModel = BaseModel
        self.Z_Info = Z_Info
        self.CurrentNodeObjValue = None
        self.LC = None
        self.RC = None
        self.VarWillBranch = None

        Z = self.CurrentCOModel.getVariable('Z')

        print('\n当前结点约束:', [self.Z_Info[i] if i in self.Z_Info.keys() else None for i in range(1, 1 + M)])

        if self.Z_Info:  # 有Z_Info
            # 加入新约束
            if len(self.Z_Info) == 1:
                location = list(self.Z_Info.keys())[0]
                NEW_Z = Z.index(location - 1)
                b_tmp = float(self.Z_Info[location])
            else:
                NEW_Z = Expr.vstack([Z.index(i - 1) for i in self.Z_Info.keys()])
                b_tmp = [float(self.Z_Info[i]) for i in self.Z_Info.keys()]
            new_constr = self.CurrentCOModel.constraint(NEW_Z, Domain.equalsTo(b_tmp))

        self.CurrentCOModel.solve()

        print('当前结点的解 Z:', np.round(self.CurrentCOModel.getVariable('Z').level(), 2))
        print('当前结点的解 I:', np.round(self.CurrentCOModel.getVariable('I').level(), 2))
        print('当前结点目标函数值:', self.CurrentCOModel.primalObjValue(), '----已知的最优可行解值:', BESTFEASIBLEVALUE)

        if self.Z_Info:  # 去掉新约束
            new_constr.update(Expr.mul(0.0, NEW_Z))
            if len(self.Z_Info) == 1:
                new_constr.update([b_tmp])
            else:
                new_constr.update(b_tmp)

    def GenerateChild(self):
        # declare global variables, then this method will use the global values
        global BESTFEASIBLEVALUE, Z_BESTFEASIBLESOLUTION, I_BESTFEASIBLESOLUTION

        if self.CurrentCOModel.getProblemStatus() == ProblemStatus.PrimalAndDualFeasible:
            self.CurrentNodeObjValue = self.CurrentCOModel.primalObjValue()
            alpha = self.CurrentCOModel.getVariable('Alpha').level()
            z = self.CurrentCOModel.getVariable('Z').level()
            inv = self.CurrentCOModel.getVariable('I').level()

        else:
            print(self.CurrentCOModel.getProblemStatus())
            print('-----------\n', 'Model Infeasible\n', '-----------\n')
            self.LC = 'STOP'
            self.RC = 'STOP'

        # 若当前有可行解
        if self.CurrentNodeObjValue != None:

            # 更新bound
            if self.CurrentNodeObjValue >= BESTFEASIBLEVALUE or all(abs(z * z - z) <= 1.0e-4):
                # 当前解还不如某个feasible solution: Cut branch
                # or 当前解已经是整数解了
                self.LC = 'STOP'
                self.RC = 'STOP'

                if self.CurrentNodeObjValue < BESTFEASIBLEVALUE:  # current solution is better, then update
                    BESTFEASIBLEVALUE = self.CurrentNodeObjValue
                    Z_BESTFEASIBLESOLUTION = np.round(z)
                    I_BESTFEASIBLESOLUTION = np.round(inv, 2)

            # 把已经是整数的Z存起来，给下一个结点的Z_Info
            Z_Info_child = copy.deepcopy(self.Z_Info)
            for i in range(1, 1 + M):
                if abs(z[i - 1] ** 2 - z[i - 1]) <= 1.0e-4:
                    Z_Info_child.update({i: round(z[i - 1])})

        if len(Z_Info_child) != M and self.LC != 'STOP' and self.RC != 'STOP':
            # Heuristically 去选 Alpha中道路的price负的最大的warehouse node，
            # 说明从这个仓库出去的物资能够最大程度降低第二阶段原max目标函数
            road_value = alpha[:r] + alpha[-r:]
            road_value = {ROAD[k]: abs(road_value[k]) for k, road in enumerate(ROAD)}
            location_value = [(i, sum(value if road[0] == i else 0 for road, value in road_value.items()))
                              for i in range(1, 1 + M)]
            sorted_location_value = sorted(location_value, reverse=True, key=lambda pair: pair[1])

            for location, value in sorted_location_value:
                if location not in Z_Info_child.keys():
                    self.VarWillBranch = location
                    break

            print('分枝变量:', self.VarWillBranch)

            # left child
            Z_Info_lc = {**Z_Info_child, **{location: 0}}
            self.LC = BranchBoundNode(self.CurrentCOModel, Z_Info_lc)
            self.LC.GenerateChild()

            # right child
            Z_Info_rc = {**Z_Info_child, **{location: 1}}
            self.RC = BranchBoundNode(self.CurrentCOModel, Z_Info_rc)
            self.RC.GenerateChild()


# initial feasible solution 选择MU_d+SIGMA_d 最大的 M/2个仓库
max_demand = MU_d + SIGMA_d
location_max_demand = np.asarray(
    [sum(max_demand[j - 1] if ori == i else 0 for ori, des in ROAD) for i in range(1, 1 + M)])
largest_halfM_loc = location_max_demand.argsort()[int(-M / 2):][::-1]

initial_Z_Info = {loc + 1: 1 for loc, value in enumerate(location_max_demand) if loc in largest_halfM_loc}
initial_Z_Info.update({loc + 1: 0 for loc, value in enumerate(location_max_demand) if loc not in largest_halfM_loc})
initialModel = GetSolvedCOModel(Z_Info=initial_Z_Info)

BESTFEASIBLEVALUE = initialModel.primalObjValue()
Z_BESTFEASIBLESOLUTION = [initial_Z_Info[loc] for loc in range(1, 1 + M)]
I_BESTFEASIBLESOLUTION = list(np.round(initialModel.getVariable('I').level(), 2))

# Tree
root = BranchBoundNode(GetSolvedCOModel(), {})
root.GenerateChild()
