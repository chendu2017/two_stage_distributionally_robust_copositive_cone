from typing import List, Dict
from simulator.Simulator import Simulator
import numpy as np


class Experiment(object):
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
        self.demand_realizations = experiment_params['demand_realizations']

        # first- & second-moment
        self.mu_sample = np.asarray([d_r for k, d_r in self.demand_realizations.items()]).mean(axis=0).tolist()
        self.sigma_sample = np.asarray([np.outer(d_r, d_r)
                                        for k, d_r in self.demand_realizations.items()]).mean(axis=0).tolist()

        # model
        self.model_co = None
        self.model_cp_2s = None
        self.model_ldr = None

        # Simulator
        self.simulator = Simulator(self.m, self.n, self.graph)

    def Run_Co_Model(self, co_params):
        from co.CO_Model import COModel
        co_model = COModel(self.m, self.n, self.f, self.h, self.mu_sample, self.sigma_sample, self.graph, co_params)
        co_model.Build_Co_Model()
        co_model = co_model.Solve_Co_Model()
        self.model_co = co_model

    def Run_LDR_Model(self, ldr_params):
        import ldr.LDRModel as LDRModel
        ldr_model = LDRModel.Build_LDR_Model(self.m, self.n, self.f, self.h, self.mu_sample, self.sigma_sample,
                                             self.graph, ldr_params)
        ldr_model = ldr_model.Solve_LDR_model()
        self.model_ldr = ldr_model

    def Simulate_in_Sample(self, sol):
        self.simulator.setSol(sol)
        self.simulator.setDemand_Realizations(self.demand_realizations)
        results = self.simulator.Run_Simulations()
        return results

    def Simulate_out_Sample(self, sol, d_rs: Dict[int, List[float]]):
        self.simulator.setSol(sol)
        self.simulator.setDemand_Realizations(d_rs)
        results = self.simulator.Run_Simulations()
        return results


if  __name__ == '__main__':
    from test_example.four_by_four_d_rs import m, n, f, h, d_rs, graph

    e_params = {'m': m,
                'n': n,
                'f': f,
                'h': h,
                'demand_realizations': d_rs,
                'graph': graph}

    e = Experiment(e_params)
    co_params = {}
    e.Run_Co_Model(co_params)
    co_model = e.model_co
    print(co_model.primalObjValue())
    print('I:', co_model.getVariable('I').level())
    print('Z:', co_model.getVariable('Z').level())
















