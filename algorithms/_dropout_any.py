import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from algorithms import BO, SA, EA, Random
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy
from ._utils import get_init_samples, select, get_subset

log = logging.getLogger(__name__)


class DropoutAny(BaseOptimizer):
    def __init__(self, dims, lb, ub, active_dims, inner_opt_type, fillin_type, reset_freq, **config):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.active_dims = active_dims
        self.inner_opt_type = inner_opt_type
        self.reset_freq = reset_freq
        self.config = config

        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]

        self.train_X = []
        self.train_Y = []

        self._reset_inner_opt()

    def _select(self, dims, active_dims):
        idx = select(dims, active_dims)
        return idx

    def _get_inner_opt(self, inner_opt_type):
        inner_opt_class = {
            'bo': BO,
            'sa': SA,
            'ea': EA,
            'random': Random,
        }
        inner_opt = inner_opt_class[inner_opt_type](dims=self.dims, lb=self.lb, ub=self.ub, **self.config)
        return inner_opt

    def ask(self) -> List[np.ndarray]:
        if len(self.train_X) == 0:
            new_X = get_init_samples('permutation', 1, self.dims, self.lb, self.ub)
        else:
            proposed_X = self.inner_opt.ask()

            # fill in
            new_X = []
            for i in range(len(proposed_X)):
                fixed_vars = {j: pos for j, pos in zip(self.idx, proposed_X[i])}
                new_x = self.fillin_strategy.fillin(fixed_vars)
                new_X.append(new_x)
        
        return new_X

    def tell(self, X: List[np.ndarray], Y: List):
        self.inner_opt.tell(X, Y)

        for x, y in zip(X, Y):
            self.fillin_strategy.update(x.reshape(1, -1), y)

        self.cnt += 1
        if self.cnt % self.reset_freq == 0:
            self._reset_inner_opt()

    def _reset_inner_opt(self):
        self.cnt = 0
        self.idx = self._select(self.dims, self.active_dims)
        self.inner_opt = self._get_inner_opt(self.inner_opt_type)

        if len(self.train_X) > 0:
            train_X_np = np.vstack(self.train_X)
            subset_X = get_subset(train_X_np, self.idx)
            self.inner_opt.tell(subset_X, self.train_Y)
