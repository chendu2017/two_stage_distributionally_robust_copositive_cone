import numpy as np
import json

m, n = 4, 4
f, h = [10, 10, 10, 10], [0.1, 0.1, 0.1, 0.1]
graph = [[1, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 0, 1, 1]]
# demand_realizations
_ = np.random.normal(loc=[20, 20, 20, 20], scale=[5, 5, 5, 5], size=[20, n])
_[_ <= 0] = 0
d_rs = {i: _[i].tolist() for i in range(_.shape[0])}

numerical_input = json.dumps({'m': m,
                              'n': n,
                              'f': f,
                              'h': h,
                              'graph': graph,
                              'd_rs': d_rs})
with open('./inputs.txt', 'a') as f:
    f.write(numerical_input + '\n')
