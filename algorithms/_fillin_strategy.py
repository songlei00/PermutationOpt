from textwrap import fill
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, dims):
        self.dims = dims

    def init_strategy(self, X, Y):
        for x, y in zip(X, Y):
            self.update(x, y)
        
    @abstractmethod
    def fillin(self, fixed_vars, lb, ub):
        pass

    @abstractmethod
    def update(self, x, y):
        pass


class PermutationRandomStrategy(BaseStrategy):
    def fillin(self, fixed_vars: Dict[int, int], lb, ub):
        keys = [int(i) for i in fixed_vars.keys()]
        vals = [int(i) for i in fixed_vars.values()]
        fillin_vals = list(set(range(self.dims)) - set(keys))
        fillin_vals = np.random.permutation(fillin_vals)
        print('fill in', fillin_vals)
        new_x = np.zeros(self.dims)
        for k, v in fixed_vars.items():
            new_x[int(v)] = k
        i = 0
        for j in range(self.dims):
            if j not in vals:
                new_x[j] = fillin_vals[i]
                i += 1
        assert len(set(new_x)) == self.dims
        return new_x

    def update(self, x, y):
        pass