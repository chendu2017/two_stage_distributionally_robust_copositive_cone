import json


class Experiment:
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

    def BuildCoModel_Reduced(self, DRO_params):
        from COModel import BuildModel_Reduced
        co_model = BuildModel_Reduced(self.m, self.n, self.f, self.h, self.mu, self.sigma, self.graph)
        pass


















