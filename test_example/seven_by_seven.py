m, n = 7, 7
f, h = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 10.0], [0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]
first_moment = [50, 30, 50, 40, 60, 50, 30]
second_moment = [
    [2524.62950802, 1499.71677314, 2499.85680761, 2000.04481979, 2999.13922019, 2499.85146255, 1499.64753851],
    [1499.71677314, 924.78940217, 1499.8153585, 1200.03688562, 1799.40773938, 1499.82089159, 899.72681527],
    [2499.85680761, 1499.8153585, 2525.11022519, 2000.27464081, 2999.35130695, 2500.08639768, 1499.74561068],
    [2000.04481979, 1200.03688562, 2000.27464081, 1625.34086862, 2399.77270816, 2000.30005551, 1199.89938988],
    [2999.13922019, 1799.40773938, 2999.35130695, 2399.77270816, 3623.35859886, 2999.39874522, 1799.28699785],
    [2499.85146255, 1499.82089159, 2500.08639768, 2000.30005551, 2999.39874522, 2525.16764792, 1499.72540345],
    [1499.64753851, 899.72681527, 1499.74561068, 1199.89938988, 1799.28699785, 1499.72540345, 924.68110993]]
graph = [[1, 1, 0, 1, 0, 1, 1],
         [0, 1, 1, 0, 0, 0, 1],
         [1, 1, 0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1, 1, 1],
         [0, 0, 1, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0],
         [1, 1, 0, 0, 1, 0, 1]]