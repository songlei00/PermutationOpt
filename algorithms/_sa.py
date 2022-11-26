import numpy as np
from typing import List
import logging
from collections import deque
from ._base import BaseOptimizer
from ._ea_operator import swap_mutation, order_crossover
from ._utils import get_init_samples

log = logging.getLogger(__name__)


class SA(BaseOptimizer):
    def __init__(self, dims, lb, ub, decay, T, mutation_type='swap'):
        self.dims = dims
        self.decay = decay
        self.init_T = T
        self.T = T
        self.mutation_type = mutation_type
        self.lb = lb
        self.ub = ub

        self.cache_X = deque()
        self.best_x = None
        self.best_y = None

    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points

    def _mutation(self, x):
        if self.mutation_type == 'swap':
            next_x = swap_mutation(x)
        else:
            raise NotImplementedError
        return next_x

    def ask(self) -> List[np.ndarray]:
        if self.best_x is None:
            points = self._init_samples('permutation', 10)
            self.cache_X.extend(points)

        if len(self.cache_X) > 0:
            return [self.cache_X.popleft()]

        x = self._mutation(self.best_x)
        return [x]

    def tell(self, X: List[np.ndarray], Y: List):
        if self.best_x is None:
            self.best_x = X[0]
            self.best_y = Y[0]

        # simulated annealing
        for x, y in zip(X, Y):
            if y > self.best_y:
                self.best_x = x 
                self.best_y = y 
            else:
                probability = np.exp(-(self.best_y - y) / self.T)
                if np.random.uniform(0, 1) < probability:
                    self.best_x = x
                    self.best_y = y 

        self.T = self.decay * self.T