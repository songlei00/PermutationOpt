# from benchmarks.CONOPLib_t62_s7.call import EarthBenchmark

# soln = """18,  74,  26,  56, 112,  72, 118,  88,  34,  12, 122,  82,  24,
#         68,  40,  10,   4,   2,  16,   8,   6,  13,  62,  84,  90,  42,
#         54,  50,  44,  32,  64,  30,  48, 108,  52,  98, 104, 102, 116,
#        114, 119,  60,  58,  28,  22,  38,  73,  78,  80, 100,   0,  70,
#         96,  20,  46,  76,  66,  36,  75,  94,  92,  91,  14,  41,  86,
#        106,  97,  63, 101,  95,  59, 113, 123,  85,  93,  17,  27,  21,
#        103,  53, 110,  71,  83,  55,  19,  51, 111,  25,  77, 120,  89,
#          5,  61, 121,  57,  99,  15,  67,  87, 109,  35,  49,   9, 107,
#          1,  81,  47, 105,  43,  29,   7,  23,  33,  69,  39,  45,  31,
#        115,  79,   3, 117,  65,  11,  37"""
# benchmark = EarthBenchmark()
# soln = [int(i) for i in soln.split(',')]
# print(benchmark(soln))


import numpy as np
from benchmarks import TSPProblem

dims = 48
tsp = TSPProblem('./benchmarks/tsp_data/att48', is_load_opt=True)
print('opt length: {}'.format(tsp.opt_len))
print('opt solution: {}'.format(tsp.opt))

print(tsp.opt + 1)
print(tsp((tsp.opt + 1) % 48))

# 对方差做测试
# n = 30
# fixed_idx = range(5)
# random_data = [np.random.permutation(dims) for _ in range(n)]
# choices = list(set(range(dims)) - set(tsp.opt[fixed_idx]))
# print(choices)
# fixed_data = [np.hstack([tsp.opt[fixed_idx], np.random.permutation(choices)]) for _ in range(n)]

# random_data_Y = [tsp(data) for data in random_data]
# fixed_data_Y = [tsp(data) for data in fixed_data]

# all_data = random_data + fixed_data
# all_Y = random_data_Y + fixed_data_Y
# data_std = np.std(all_data, axis=0)

# print('random mean Y: {}'.format(np.mean(random_data_Y)))
# print('fixed mean Y: {}'.format(np.mean(fixed_data_Y)))
# print(data_std)
# print(data_std.argsort())

# random_std = np.std(random_data, axis=0)
# fix_std = np.std(fixed_data, axis=0)
# print(fix_std)

# all_data_np = np.array(all_data)

# idx = np.argsort(all_Y)[-10: ]
# print(np.std(all_data_np[idx], axis=0))


# 其他测试
n = 200
dims = 48
random_data = np.vstack([np.random.permutation(dims) for _ in range(n)])
Y = np.hstack([tsp(random_data[i]) for i in range(len(random_data))])
idx = np.argsort(Y)
bad_idx = idx[: n//2]
good_idx = idx[n//2: ]
good_data, good_Y = random_data[good_idx], Y[good_idx]
bad_data, bad_Y = random_data[bad_idx], Y[bad_idx]

print(good_data)

print(np.mean(good_Y))
print(np.mean(bad_Y))
