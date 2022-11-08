from textwrap import fill
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Dict


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, dims, lb, ub):
        self.dims = dims
        self.lb = lb
        self.ub = ub

    def init_strategy(self, X, Y):
        for x, y in zip(X, Y):
            self.update(x, y)
        
    @abstractmethod
    def fillin(self, fixed_vars: Dict[int, int]):
        pass

    @abstractmethod
    def update(self, x, y):
        pass


class PermutationRandomStrategy(BaseStrategy):
    def fillin(self, fixed_vars: Dict[int, int]):
        keys = [int(i) for i in fixed_vars.keys()]
        vals = [int(i) for i in fixed_vars.values()]
        fillin_vals = list(set(range(self.dims)) - set(keys))
        fillin_vals = np.random.permutation(fillin_vals)
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


class PermutationBestKStrategy(BaseStrategy):
    def __init__(self, dims, lb, ub, k):
        BaseStrategy.__init__(self, dims, lb, ub)
        self.best_xs = np.zeros((0, dims))
        self.best_ys = np.zeros(0)
        self.k = k

    def update(self, x, y):
        if len(self.best_xs) < self.k:
            self.best_xs = np.vstack((self.best_xs, x))
            self.best_ys = np.vstack((self.best_ys, y))
        else:
            min_y_idx = np.argmin(self.best_ys)
            if y > self.best_ys[min_y_idx]:
                self.best_xs[min_y_idx] = x
                self.best_ys[min_y_idx] = y
        assert len(self.best_xs) <= self.k


class PermutationBestKPosStrategy(PermutationBestKStrategy):
    def fillin(self, fixed_vars: Dict[int, int]):
        curr_idx = np.array(fixed_vars.keys())
        curr_pos = np.array(fixed_vars.values())


class PermutationBestKOrderStrategy(PermutationBestKStrategy):
    def fillin(self, fixed_vars: Dict[int, int]):
        pass