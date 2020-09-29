import json


class Experiment(object):
    def __init__(self, experiment_params):
        experiment_params = json.loads(experiment_params)

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

        # first-moment of demand
        assert len(experiment_params['mu']) == self.n, 'dimensions (\mu & n) not match'
        self.mu = experiment_params['mu']

        # second-moment of demand
        assert len(experiment_params['sigma']) == self.n, 'dimensions (\sigma & n) not match'
        self.sigma = experiment_params['sigma']

        # initial graph
        self.graph = experiment_params['graph']

    def Run_DRO_Model(self, DRO_params):
        from co.COModel import Build_CoModel, Solve_CoModel
        co_model = Build_CoModel(self.m, self.n, self.f, self.h, self.mu, self.sigma, self.graph, DRO_params)
        co_model = Solve_CoModel(co_model)
        return co_model


if  __name__ == '__main__':
    from test_example.six_by_six import m, n, f, h, first_moment, second_moment, graph

    e_params = {'m': m,
                'n': n,
                'f': f,
                'h': h,
                'mu': first_moment,
                'sigma': second_moment,
                'graph': graph}
    e_params = json.dumps(e_params)

    e = Experiment(e_params)
    DRO_params = {}
    co_model = e.Run_DRO_Model(DRO_params)
    print(co_model.primalObjValue())
















