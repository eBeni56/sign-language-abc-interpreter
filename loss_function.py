import numpy as np
from abc import ABC, abstractmethod

np.seterr(divide='raise', invalid='raise', over='raise', under='ignore')

class LossFunction(ABC):
    @abstractmethod
    def f(self, predictions, targets):
        pass

    @abstractmethod
    def df(self, predictions, targets):
        pass


class MeanSquaredError(LossFunction):
    def f(self, predictions, targets):
        return np.sum((predictions - targets)**2) / predictions.shape[0]

    def df(self, predictions, targets):
        return 2 * (predictions - targets) / predictions.shape[0]


class CategorialCrossEntropy(LossFunction):
    def f(self, predictions, targets):

        # print(-np.sum(targets * np.log(predictions)))
        try:
            return -np.sum(targets * np.log(predictions))
        except FloatingPointError:
            print("Targets:", targets)
            print("Predictions: ", predictions)
            raise ValueError("error")
    
    def df(self, predictions, targets):
        return -targets / predictions  


class BinaryCrossEntropy(LossFunction):
    def f(self, predictions, targets):
        return -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)) / predictions.shape[0] 

    def df(self, predictions, targets):
        return (predictions - targets) / (predictions * (1 - predictions)) 

