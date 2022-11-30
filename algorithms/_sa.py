import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from ._ea_operator import swap_mutation
from ._utils import get_init_samples

log = logging.getLogger(__name__)


class SA(BaseOptimizer):
    def __init__(self, dims, lb, ub, decay, T, update_freq, mutation_type='swap'):
        self.dims = dims
        self.decay = decay
        self.init_T = T
        self.T = T
        self.update_freq = update_freq
        self.mutation_type = mutation_type
        self.lb = lb
        self.ub = ub

        self.best_x = None
        self.best_y = None
        self.cnt = 0

    def _mutation(self, x):
        if self.mutation_type == 'swap':
            next_x = swap_mutation(x)
        else:
            raise NotImplementedError
        return next_x

    def ask(self) -> List[np.ndarray]:
        if self.best_x is None:
            x = get_init_samples('permutation', 1, self.dims, self.lb, self.ub)[0]
        else:
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

        self.cnt += 1
        if self.cnt % self.update_freq == 0:
            self.cnt = 0
            self.T = self.decay * self.T