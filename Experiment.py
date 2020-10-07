import time
from typing import List, Dict

from memory_profiler import profile
from gurobipy.gurobipy import Model as grbModel
from simulator.Simulator import Simulator
import numpy as np
from deprecated import deprecated
from mosek.fusion import Model
import json
from co.CO_Model import COModel


class Experiment(object):
    model_co: Model

    def __init__(self, e_param):
        # number of warehouses
        self.m = e_param['m']

        # number of locations
        self.n = e_param['n']

        # cost modifier
        self.kappa = e_param['kappa']

        # holding cost
        assert len(e_param['h']) == self.m, 'dimensions (f & m) not match'
        self.h = [_*self.kappa for _ in e_param['h']]

        # setup cost
        assert len(e_param['f']) == self.m, 'dimensions (h & m) not match'
        self.f = [_*self.kappa for _ in e_param['f']]

        # mu, sigma
        self.mu = e_param['mu']
        self.sigma = e_param['sigma']

        # cv, rho
        self.cv = e_param['cv']
        self.rho = e_param['rho']

        # initial graph
        self.graph = e_param['graph']

        # demand realizations for simulation
        self.d_rs = {int(k): d_r for k, d_r in e_param['d_rs'].items()}

        # model
        self.model_co = None
        self.model_ldr = None
        self.model_saa = None
        self.model_mv = None
        self.model_mv_node = 0

        # cpu time
        INF = float('inf')
        self.co_time = INF
        self.co_node = 0
        self.mv_time = INF
        self.mv_node = 0
        self.saa_time = INF

        # Simulator
        self.simulator = Simulator(self.m, self.n, self.graph)

    def Run_Co_Model(self, co_param=None) -> Model:
        # Run_Co_Model will be called twice. Init co_node as 0 to remove former running result
        self.co_node = 0
        CO_model = COModel(self.m, self.n, self.f, self.h, self.mu, self.sigma, self.graph, co_param)
        # record solving time
        start = time.perf_counter()
        self.model_co, self.co_node = CO_model.Solve_Co_Model()
        self.co_time = time.perf_counter() - start
        return self.model_co

    def Run_MV_Model(self, mv_params=None) -> Model:
        """
        MV_model is absolutely the same as CO_model except the second-moment matrix construction
        Therefore, we re-use COModel class, and modify the second-moment matrix manually.
        :param mv_params:
        """
        from co.CO_Model import COModel
        sigma_mv = np.outer(self.mu, self.mu) + np.diag([(single_mu*self.cv)**2 for single_mu in self.mu])
        mv_model = COModel(self.m, self.n, self.f, self.h, self.mu, sigma_mv, self.graph, mv_params)
        mv_model.Build_Co_Model()
        # record solving time
        start = time.perf_counter()
        self.model_mv, self.mv_node = mv_model.Solve_Co_Model()
        self.mv_time = time.perf_counter() - start
        return self.model_mv

    def Run_SAA_Model(self, saa_param=None) -> grbModel:
        from benchmark.SAA_Model import SAAModel
        saa_model = SAAModel(self.m, self.n, self.f, self.h, self.graph, saa_param)
        # record solving time
        start = time.perf_counter()
        saa_model = saa_model.SolveStoModel()
        self.saa_time = time.perf_counter() - start
        self.model_saa = saa_model
        return saa_model

    @deprecated(reason='LDR returns all zero coefficients')
    def Run_LDR_Model(self, ldr_params):
        import ldr.LDRModel as LDRModel
        ldr_model = LDRModel.Build_LDR_Model(self.m, self.n, self.f, self.h, self.mu_sample, self.sigma_sample,
                                             self.graph, ldr_params)
        ldr_model = ldr_model.Solve_LDR_model()
        self.model_ldr = ldr_model
        return ldr_model

    def Simulate_Second_Stage(self, sol):
        self.simulator.setSol(sol)
        self.simulator.setDemand_Realizations(self.d_rs)
        results = self.simulator.Run_Simulations()
        return results


if __name__ == '__main__':
    from test_example.four_by_four_d_rs import m, e_param, saa_param, co_param

    print(e_param)
    print(saa_param)
    e = Experiment(e_param)

    # saa_model
    saa_model = e.Run_SAA_Model(saa_param)
    print(saa_model.ObjVal)
    print('I:', [saa_model.getVarByName(f'I[{i}]').x for i in range(m)])
    print('Z:', [saa_model.getVarByName(f'Z[{i}]').x for i in range(m)])

    # co_model
    co_model = e.Run_Co_Model(co_param)
    print(co_model.primalObjValue())
    print('I:', co_model.getVariable('I').level())
    print('Z:', co_model.getVariable('Z').level())

    # mv_model
    mv_model = e.Run_MV_Model(co_param)
    print(mv_model.primalObjValue())
    print('I:', mv_model.getVariable('I').level())
    print('Z:', mv_model.getVariable('Z').level())















