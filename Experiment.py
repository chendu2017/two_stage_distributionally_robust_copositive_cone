from typing import List, Dict
from simulator.Simulator import Simulator
import numpy as np
from deprecated import deprecated
from mosek.fusion import Model
import json

class Experiment(object):
    model_co: Model

    def __init__(self, experiment_params):
        # number of warehouses
        self.m = experiment_params['m']

        # number of locations
        self.n = experiment_params['n']

        # holding cost
        assert len(experiment_params['h']) == self.m, 'dimensions (f & m) not match'
        self.h = experiment_params['h']

        # setup cost
        assert len(experiment_params['f']) == self.m, 'dimensions (h & m) not match'
        self.f = experiment_params['f']

        # initial graph
        self.graph = experiment_params['graph']

        # in-sample demand realization
        self.d_rs = experiment_params['d_rs']

        # first- & second-moment
        self.mu_sample = np.asarray([d_r for k, d_r in self.d_rs.items()]).mean(axis=0).tolist()
        self.sigma_sample = np.asarray([np.outer(d_r, d_r)
                                        for k, d_r in self.d_rs.items()]).mean(axis=0).tolist()
        self.var_sample = np.asarray([d_r for k, d_r in self.d_rs.items()]).var(axis=0).tolist()

        # model
        self.model_co = None
        self.model_cp_2s = None
        self.model_ldr = None
        self.model_saa = None
        self.model_mv = None

        # Simulator
        self.simulator = Simulator(self.m, self.n, self.graph)

    def Run_Co_Model(self, co_params=None) -> Model:
        from co.CO_Model import COModel
        co_model = COModel(self.m, self.n, self.f, self.h, self.mu_sample, self.sigma_sample, self.graph, co_params)
        co_model.Build_Co_Model()
        co_model = co_model.Solve_Co_Model()
        self.model_co = co_model
        return co_model

    def Run_MV_Model(self, mv_params=None):
        """
        MV_model is absolutely the same as CO_model except the second-moment matrix construction
        Therefore, we re-use COModel class, and modify the second-moment matrix manually.
        :param mv_params:
        """
        from co.CO_Model import COModel
        sigma_mv_sample = (np.asarray(self.sigma_sample) + np.diag(self.var_sample)).tolist()
        mv_model = COModel(self.m, self.n, self.f, self.h, self.mu_sample, sigma_mv_sample, self.graph, mv_params)
        mv_model.Build_Co_Model()
        mv_model  = mv_model.Solve_Co_Model()
        self.model_mv = mv_model
        return mv_model

    def Run_SAA_Model(self, saa_params=None):
        from benchmark.SAA_Model import SAAModel
        saa_model = SAAModel(self.m, self.n, self.f, self.h, self.d_rs, self.graph, saa_params).SolveStoModel()
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

    def Simulate_in_Sample(self, sol):
        self.simulator.setSol(sol)
        self.simulator.setDemand_Realizations(self.d_rs)
        results = self.simulator.Run_Simulations()
        return results

    def Simulate_out_Sample(self, sol, d_rs: Dict[int, List[float]]):
        self.simulator.setSol(sol)
        self.simulator.setDemand_Realizations(d_rs)
        results = self.simulator.Run_Simulations()
        return results


if __name__ == '__main__':
    from test_example.four_by_four_d_rs import m, n, f, h, d_rs, graph

    e_params = {'m': m,
                'n': n,
                'f': f,
                'h': h,
                'graph': graph,
                'd_rs': d_rs,
                }
    print(e_params)
    co_params = {'speedup': {'Tau': False, 'Eta': False, 'W': False},
                 'bb_params': {'find_init_z': 'v1',
                               'select_branching_pos': 'v1'}}
    e = Experiment(e_params)

    # saa_model
    saa_model = e.Run_SAA_Model()
    print(saa_model.ObjVal)
    print('I:', [saa_model.getVarByName(f'I[{i}]').x for i in range(4)])
    print('Z:', [saa_model.getVarByName(f'Z[{i}]').x for i in range(4)])

    # co_model
    co_model = e.Run_Co_Model(co_params)
    print(co_model.primalObjValue())
    print('I:', co_model.getVariable('I').level())
    print('Z:', co_model.getVariable('Z').level())

    # mv_model
    mv_model = e.Run_MV_Model(co_params)
    print(mv_model.primalObjValue())
    print('I:', mv_model.getVariable('I').level())
    print('Z:', mv_model.getVariable('Z').level())















