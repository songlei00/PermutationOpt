import numpy as np
from typing import List
import logging
from ._base import BaseOptimizer
from ._utils import get_init_samples
from ._ea_operator import swap_mutation, order_crossover

log = logging.getLogger(__name__)


class EA(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, pop_size=20, init_sampler_type='permutation',
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
    
    def tell(self, X: List[np.ndarray], Y: List):
        if len(self.population) == 0:
            self.population = X
            self.fitness = Y
        else:
            print(len(self.population), len(X))
            self.population, self.fitness = self._selection(self.population, self.fitness, X, Y)