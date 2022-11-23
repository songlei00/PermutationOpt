from textwrap import fill
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Dict
from ._utils import get_subset


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
        
    def _get_fillin(self, fixed_vars, x):
        ret = np.zeros(self.dims)
        for k, v in fixed_vars.items():
            ret[int(v)] = k
        i = 0
        # print('fixed vars: {}'.format(fixed_vars))
        # print('x: {}'.format(x))
        for j in range(self.dims):
            if j not in fixed_vars.values():
                while x[i] in fixed_vars.keys():
                    i += 1
                ret[j] = x[i]
                i += 1
        # print('ret: {}'.format(ret))
        assert len(set(ret)) == self.dims
        return ret

    def update(self, x, y):
        # print('best xs: {}'.format(self.best_xs.shape))
        # print('x: {}'.format(x.shape))
        if len(self.best_xs) < self.k:
            self.best_xs = np.vstack((self.best_xs, x))
            self.best_ys = np.hstack((self.best_ys, y))
        else:
            min_y_idx = np.argmin(self.best_ys)
            if y > self.best_ys[min_y_idx]:
                self.best_xs[min_y_idx] = x
                self.best_ys[min_y_idx] = y
        assert len(self.best_xs) <= self.k


class PermutationBestKPosStrategy(PermutationBestKStrategy):
    def fillin(self, fixed_vars: Dict[int, int]):
        curr_idx = np.array(list(fixed_vars.keys()))
        curr_pos = np.array(list(fixed_vars.values()))
        subset_X = get_subset(self.best_xs, curr_idx)
        # print('subset X: {}'.format(subset_X.shape))
        # print('curr pos: {}'.format(curr_pos.shape))
        dis = np.mean((subset_X - curr_pos)**2, axis=1)
        idx = np.argmin(dis)
        fillin_x = self.best_xs[idx]
        ret = self._get_fillin(fixed_vars, fillin_x)
        return ret


class PermutationBestKOrderStrategy(PermutationBestKStrategy):
    def fillin(self, fixed_vars: Dict[int, int]):
        pass