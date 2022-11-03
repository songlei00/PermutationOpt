from abc import ABCMeta, abstractmethod


class BaseOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self):
        pass
    
    @abstractmethod
    def tell(self, x, y):
        pass