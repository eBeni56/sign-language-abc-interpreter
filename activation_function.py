import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def df(self, x):
        pass


class ReLU(ActivationFunction):
    def f(self, x):
        return np.maximum(0, x)
    
    def df(self, x):
        return np.where(x > 0, 1, 0)
    

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def f(self, x):
        return np.maximum(self.alpha * x, x)

    def df(self, x):
        return np.where(x > 0, 1, self.alpha) 
 


class Sigmoid(ActivationFunction):
    def f(self, x):
        return 1 / (1 + np.exp(-x))
    

    def df(self, x):
        value = self.f(x)
        return value * (1 - value)


class Tanh(ActivationFunction):
    def f(self, x):
        value = np.exp(x)
        value_recip = 1 / value
        return  (value - value_recip) / (value + value_recip)

    def df(self, x):
        return 1 - self.f(x)**2


class SoftPlus(ActivationFunction):
    def f(self, x):
        return np.log(1 + np.exp(x))
    
    def df(self, x):
        value = np.exp(x)
        return value / (1 + value)


class SoftMax(ActivationFunction):
    def f(self, x):
        m = np.max(x)
        expo = np.exp(x - m)
        return expo / np.sum(expo)

    def df(self, x):
        raise ValueError("Not implemented")