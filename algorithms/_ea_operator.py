import numpy as np
import logging

log = logging.getLogger(__name__)


def swap_mutation(x: np.ndarray, repeats=1):
    assert x.ndim == 1
    next_x = x.copy()
    for _ in range(repeats):
        i, j = np.random.choice(range(len(x)), 2, replace=False)
        next_x[i], next_x[j] = next_x[j], next_x[i]
    return next_x


def order_crossover(x1: np.ndarray, x2: np.ndarray):
    assert x1.ndim == 1 and x2.ndim == 1
    x_len = len(x1)
    next_x = np.zeros(x_len)
    
    i, j = np.random.choice(range(x_len), 2, replace=False)
    i, j = min(i, j), max(i, j)
    next_x[i: j] = x1[i: j]
    copy_idx = j
    for k in range(j, j+x_len-(j-i)):
        while x2[copy_idx] in next_x[i: j]:
            copy_idx = (copy_idx + 1) % x_len
        next_x[k % x_len] = x2[copy_idx]
        copy_idx = (copy_idx + 1) % x_len
    log.debug('next x: {}'.format(next_x))
    return next_x