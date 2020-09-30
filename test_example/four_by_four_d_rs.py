import numpy as np
m, n = 4, 4
f, h = [10, 12, 13, 11], [0.5, 0.2, 0.3, 0.4]
graph = [[1, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 0, 1, 1]]
# demand_realizations
_ = np.random.normal(loc=[20, 15, 25, 30], scale=[5, 5, 5, 5], size=[1000, n])
_[_ <= 0] = 0
d_rs = {i: _[i].tolist() for i in range(_.shape[0])}


