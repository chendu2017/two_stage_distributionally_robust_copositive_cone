class MVModel(object):
    def __init__(self, m, n, f, h, mu, sigma, graph, mv_params=None):
        self.m, self.n = m, n
        self.f, self.h = f, h
        self.mu, self.sigma = mu, sigma
        self.graph = graph
        self.mv_params = mv_params
        self.model = None

        self.roads = [(i, j) for i in range(m) for j in range(n) if graph[i][j] == 1]
