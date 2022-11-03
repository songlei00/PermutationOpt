import numpy as np
from collections import deque


def swap_mutation():
    pass


class EA:
    def __init__(self):
        self.train_X = []
        self.train_y = []
        self.pop = []
    
    def ask(self):
        if len(self.pop) == 0:
            self.pop = init()
        else:
            self.offspring = []
            for _ in range(self.offspring_size):
                self.crossover()
                self.mutate()
                
        pass
    
    def tell(self):
        pass