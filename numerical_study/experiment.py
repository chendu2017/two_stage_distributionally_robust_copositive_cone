import time

import numpy as np
from deprecated import deprecated
from gurobipy.gurobipy import Model as grbModel
from mosek.fusion import Model

from co.Bootstrapping import Bootstrap
from co.CO_Model import COModel
from numerical_study.ns_utils import Generate_d_rs, Calculate_True_Cov_Matrix, Calculate_True_Sigma_Matrix, \
    Calculate_Sampled_Sigma_Matrix, Calculate_Sampled_Cov_Matrix
from simulator.Simulator import Simulator


class Experiment(object):
    model_co: Model

    def __init__(self, e_param):
        self.e_param = e_param

        # number of warehouses, locations
        self.m, self.n = e_param['m'], e_param['n']

        # cost modifier
        self.kappa, self.cv, self.rho = e_param['kappa'], e_param['cv'], e_param['rho']

        # holding cost
        assert len(e_param['h']) == self.m, 'dimensions (f & m) not match'
        self.h = np.asarray([_ * self.kappa for _ in e_param['h']])

        # setup cost
        assert len(e_param['f']) == self.m, 'dimensions (h & m) not match'
        self.f = np.asarray([_ * self.kappa for _ in e_param['f']])

        # mu, sigma
        self.mu = np.asarray(e_param['mu'])
        self.std = np.asarray([mu * self.cv for mu in self.mu])
        self.sigma = Calculate_True_Sigma_Matrix(self.mu, self.cv, self.rho)
        self.cov = Calculate_True_Cov_Matrix(self.mu, self.cv, self.rho)

        # initial graph
        self.graph = e_param['graph']

        # demand_observations
        self.demand_observations = Generate_d_rs(self.mu, self.cv, self.rho,
                                                 sample_size=e_param['demand_observations_sample_size'],
                                                 dist=e_param['demand_observations_dist'],
                                                 seed=e_param['seed'])
        self.e_param['demand_observations'] = self.demand_observations.tolist()
        self.mu_sampled = np.mean(self.demand_observations, axis=0)
        self.std_sampled = np.std(self.demand_observations, axis=0)
        self.sigma_sampled = Calculate_Sampled_Sigma_Matrix(self.demand_observations)
        self.cov_sampled = Calculate_Sampled_Cov_Matrix(self.demand_observations)

        # demand realizations for simulation
        self.d_rs_insample = Generate_d_rs(self.mu, self.cv, self.rho,
                                           sample_size=e_param['in_sample_demand_sample_size'],
                                           dist=e_param['in_sample_demand_dist'],
                                           seed=e_param['seed'])
        self.d_rs_outsample = Generate_d_rs(self.mu, self.cv, self.rho,
                                            sample_size=e_param['out_sample_demand_sample_size'],
                                            dist=e_param['out_sample_demand_dist'],
                                            seed=e_param['seed'])

        # bootstrap results
        self.mu_sampled_lb = None
        self.mu_sampled_ub = None
        self.sigma_sampled_lb = None
        self.sigma_sampled_ub = None

        # model
        self.model_co = None
        self.model_ldr = None
        self.model_saa = None
        self.model_mv = None
        self.model_det = None
        self.model_mv_node = 0

        # cpu time
        INF = float('inf')
        self.co_time = INF
        self.co_node = 0
        self.mv_time = INF
        self.mv_node = 0
        self.saa_time = INF
        self.det_time = INF

        # Simulator
        self.simulator = Simulator(self.m, self.n, self.graph)

    def Run_Co_Model(self, co_param=None) -> Model:
        # bootstrap to find out CI on sample_firstMoments and sample_secondMoments
        if co_param['bootstrap_CI'] > 0:
            IS_BOOTSTRAPPED_FLAG = True
            self.mu_sampled_lb, self.mu_sampled_ub, self.sigma_sampled_lb, self.sigma_sampled_ub = \
                Bootstrap(self.demand_observations,
                          replicates=co_param['replicates'],
                          CI=co_param['bootstrap_CI'])
            co_param.update({'mu_sampled_lb': self.mu_sampled_lb.tolist(),
                             'mu_sampled_ub': self.mu_sampled_ub.tolist(),
                             'sigma_sampled_lb': self.sigma_sampled_lb.tolist(),
                             'sigma_sampled_ub': self.sigma_sampled_ub.tolist()})
        else:
            IS_BOOTSTRAPPED_FLAG = False

        self.co_node = 0
        CO_model = COModel(self.m, self.n, self.f, self.h, self.mu_sampled, self.sigma_sampled, self.graph,
                           mu_lb=self.mu_sampled_lb, mu_ub=self.mu_sampled_ub,
                           sigma_lb=self.sigma_sampled_lb, sigma_ub=self.sigma_sampled_ub,
                           co_param=co_param, bootstrap=IS_BOOTSTRAPPED_FLAG)
        # record solving time
        start = time.perf_counter()
        self.model_co, self.co_node = CO_model.Solve_Co_Model()
        self.co_time = time.perf_counter() - start
        return self.model_co

    def Run_MV_Model(self, mv_param=None) -> Model:
        """
        MV_model is absolutely the same as CO_model except the second-moment matrix construction
        Therefore, we re-use COModel class, and modify the second-moment matrix manually.
        :param mv_params:
        """
        from co.CO_Model import COModel
        sigma_sampled_lb_mv, sigma_sampled_ub_mv = None, None
        # mv's sampled sigma is sigma_smapled minus correlation terms. lb, ub follows the same rule
        # (application only if bootstrapped)
        sigma_sampled_mv = self.sigma_sampled - self.cov_sampled + np.diag(np.diag(self.cov_sampled))
        if mv_param['bootstrap_CI'] > 0:
            IS_BOOTSTRAPPED_FLAG = True
            sigma_sampled_lb_mv = self.sigma_sampled_lb - self.cov_sampled + np.diag(np.diag(self.cov_sampled))
            sigma_sampled_ub_mv = self.sigma_sampled_ub - self.cov_sampled + np.diag(np.diag(self.cov_sampled))
            mv_param.update({'mu_sampled_lb': self.mu_sampled_lb.tolist(),
                             'mu_sampled_ub': self.mu_sampled_ub.tolist(),
                             'sigma_sampled_lb': sigma_sampled_lb_mv.tolist(),
                             'sigma_sampled_ub': sigma_sampled_ub_mv.tolist()})
        else:
            IS_BOOTSTRAPPED_FLAG = False

        mv_model = COModel(self.m, self.n, self.f, self.h, self.mu_sampled, sigma_sampled_mv, self.graph,
                           mu_lb=self.mu_sampled_lb, mu_ub=self.mu_sampled_ub,
                           sigma_lb=sigma_sampled_lb_mv, sigma_ub=sigma_sampled_ub_mv,
                           co_param=mv_param, bootstrap=IS_BOOTSTRAPPED_FLAG)
        mv_model.Build_Co_Model()
        # record solving time
        start = time.perf_counter()
        self.model_mv, self.mv_node = mv_model.Solve_Co_Model()
        self.mv_time = time.perf_counter() - start
        return self.model_mv

    def Run_SAA_Model(self, saa_param=None) -> grbModel:
        from benchmark.SAA_Model import SAAModel
        saa_model = SAAModel(self.m, self.n, self.f, self.h, self.graph, self.demand_observations, saa_param)
        # record solving time
        start = time.perf_counter()
        saa_model = saa_model.SolveStoModel()
        self.saa_time = time.perf_counter() - start
        self.model_saa = saa_model
        return saa_model

    def Run_Det_Model(self, det_param):
        from benchmark.DET_Model import DETModel
        det_model = DETModel(self.m, self.n, self.f, self.h, self.graph, det_param)
        # record solving time
        start = time.perf_counter()
        det_model = det_model.SolveDetModel()
        self.det_time = time.perf_counter() - start
        self.model_det = det_model
        return det_model

    @deprecated(reason='LDR returns all zero coefficients')
    def Run_LDR_Model(self, ldr_params):
        from ldr.LDR_Model import LDRModel
        ldr_model = LDRModel(self.m, self.n, self.f, self.h, self.mu, self.sigma, self.graph, ldr_params)
        ldr_model.Build_LDR_Model()
        ldr_model = ldr_model.Solve_LDR_Model()
        self.model_ldr = ldr_model
        return ldr_model

    def Simulate_Second_Stage(self, sol):
        self.simulator.setSol(sol)
        # in-sample
        self.simulator.setDemand_Realizations(self.d_rs_insample)
        results_insample = self.simulator.Run_Simulations()
        # out-sample
        self.simulator.setDemand_Realizations(self.d_rs_outsample)
        results_outsample = self.simulator.Run_Simulations()
        return results_insample, results_outsample

    def Set_Demand_Observations(self, demand_obsers):
        self.demand_observations = demand_obsers
        self.e_param['demand_observations'] = self.demand_observations.tolist()
        self.mu_sampled = np.mean(self.demand_observations, axis=0)
        self.std_sampled = np.std(self.demand_observations, axis=0)
        self.sigma_sampled = Calculate_Sampled_Sigma_Matrix(self.demand_observations)
        self.cov_sampled = Calculate_Sampled_Cov_Matrix(self.demand_observations)


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
