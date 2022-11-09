import numpy as np
from typing import List
import logging
from ._utils import get_init_samples

log = logging.getLogger(__name__)


def swap_mutation(x: np.array, repeats=1):
    assert x.ndim == 1
    next_x = x.copy()
    for _ in range(repeats):
        i, j = np.random.choice(range(len(x)), 2, replace=False)
        next_x[i], next_x[j] = next_x[j], next_x[i]
    return next_x


def order_crossover(x1: np.array, x2: np.array):
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


class EA:
    def __init__(self, dims, lb, ub, pop_size=20, init_sampler_type='permutation',
        mutation_type='swap', crossover_type='order'
    ):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.offspring_size = self.pop_size
        self.init_sampler_type = init_sampler_type
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type

        self.population = []
        self.fitness = []

    def _init_samples(self, init_sampler_type, n) -> List[np.array]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points

    def _mutation(self, x):
        if self.mutation_type == 'swap':
            next_x = swap_mutation(x)
        else:
            raise NotImplementedError
        return next_x

    def _crossover(self, x1, x2):
        if self.crossover_type == 'order':
            next_x = order_crossover(x1, x2)
        else:
            raise NotImplementedError
        return next_x

    def _selection(self, parents, parents_fit, offspring, offspring_fit):
        all_individual = parents + offspring
        all_fitness = parents_fit + offspring_fit
        indices = np.argsort(all_fitness)[-self.pop_size: ]
        next_generation = [all_individual[idx] for idx in indices]
        next_fitness = [all_fitness[idx] for idx in indices]
        return next_generation, next_fitness
    
    def ask(self):
        offspring = []
        if len(self.population) == 0:
            offspring.extend(self._init_samples(self.init_sampler_type, self.pop_size))
        else:
            for _ in range(self.offspring_size):
                i, j = np.random.choice(range(self.pop_size), 2, replace=False)
                x1, x2 = self.population[i], self.population[j]
                next_x = self._crossover(x1, x2)
                next_x = self._mutation(next_x)
                offspring.append(next_x)
                
        return offspring
    
    def tell(self, X: List[np.array], Y):
        if len(self.population) == 0:
            self.population = X
            self.fitness = Y
        else:
            self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)