from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, layer, learning_rate):
        self.layer = layer
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self):
        pass


class GradientDescent(Optimizer):
    def __init__(self, layer, learning_rate): 
        super().__init__(layer, learning_rate)

    
    def update(self):
        self.layer.weights -= self.learning_rate * self.layer.weight_gradients
        self.layer.biases -= self.learning_rate * self.layer.bias_gradients


class GradientDescentWithMomentum(Optimizer):
    def __init__(self, layer, learning_rate, beta = 0.9):  
        super().__init__(layer, learning_rate)

        self.beta = beta 

        #same shape and dtype as the weight and bias gradient vectors
        #ewma = exponentially weighted moving average
        self.weight_gradients_ewma = np.zeros_like(self.layer.weights)
        self.bias_gradients_ewma = np.zeros_like(self.layer.biases)

   
    def update(self):
        self.weight_gradients_ewma *= self.beta
        self.weight_gradients_ewma += (1 - self.beta) * self.layer.weight_gradients

        self.bias_gradients_ewma *= self.beta 
        self.bias_gradients_ewma += (1 - self.beta) * self.layer.bias_gradients


        self.layer.weights -= self.learning_rate * self.weight_gradients_ewma
        self.layer.biases -= self.learning_rate * self.bias_gradients_ewma



# #Adaptive Gradient, so it adapts the gradient strength, depending on how much
# #a given parameter has "progressed" (how big the updates on it were)
# #while at the plain GradientDescent variants, the learning_rate is just a single constant, that is
# #the same for any parameter, here we scale the learning_rate depending on how
# #big the gradients were at a given parameter (be it weight or bias)
# #this solves that e.g. there's a weight that always has huge gradients and there's a
# #bias that has small gradients and so we'd go much more in the direction of the weight, since
# #the bigger updates happen at the weight, while the adaptive methods scale
# #the learning_rate at the given parameter and it gets a much smaller multiplier this way
# #this makes the path to the minimum much straighter
class AdaGrad(Optimizer):
    def __init__(self, layer, learning_rate, eps = 1e-8):
        super().__init__(layer, learning_rate)

        #epsilon is a small number, it's there to avoid dividing by 0
        self.eps = eps

        # same shape and dtype as the weight and bias gradient vectors
        self.weight_gradients_squared_sum = np.zeros_like(self.layer.weights)
        self.bias_gradients_squared_sum = np.zeros_like(self.layer.biases)


    def update(self):
        self.weight_gradients_squared_sum += self.layer.weight_gradients * self.layer.weight_gradients
        self.bias_gradients_squared_sum += self.layer.bias_gradients * self.layer.bias_gradients 

        weight_denominator = np.sqrt(self.weight_gradients_squared_sum) + self.eps 
        bias_denominator = np.sqrt(self.bias_gradients_squared_sum) + self.eps

        self.layer.weights -= self.layer.weight_gradients / weight_denominator * self.learning_rate
        self.layer.biases -= self.layer.bias_gradients / bias_denominator * self.learning_rate



class RMSProp(Optimizer):
    def __init__(self, layer, learning_rate, beta = 0.9, epsilon = 1e-8):
        super().__init__(layer, learning_rate)

        self.beta = beta
        self.epsilon = epsilon

        #same shape and dtype as the weight and bias gradient vectors
        #exponentially weighted moving average of squared gradients
        self.weight_gradients_squared_ewma = np.zeros_like(self.layer.weights)
        self.bias_gradients_squared_ewma = np.zeros_like(self.layer.biases)


    def update(self):
        self.weight_gradients_squared_ewma *= self.beta
        squared_weight_gradients = self.layer.weight_gradients * self.layer.weight_gradients
        squared_weight_gradients *= (1 - self.beta)
        self.weight_gradients_squared_ewma += squared_weight_gradients         

        self.bias_gradients_squared_ewma *= self.beta
        squared_bias_gradients = self.layer.bias_gradients * self.layer.bias_gradients
        squared_bias_gradients *= (1 - self.beta)
        self.bias_gradients_squared_ewma += squared_bias_gradients

        #epsilon is a small number, it's there to avoid dividing by 0
        weight_denominator = np.sqrt(self.weight_gradients_squared_ewma) + self.epsilon
        bias_denominator = np.sqrt(self.bias_gradients_squared_ewma) + self.epsilon

        self.layer.weights -= self.layer.weight_gradients / weight_denominator * self.learning_rate
        self.layer.biases -= self.layer.bias_gradients / bias_denominator * self.learning_rate




class Adam(Optimizer):
    def __init__(self, layer, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        super().__init__(layer, learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        #t is the point in time, so basically a counter for which update this is
        #we need this for the bias correction
        self.t = 0
        
        self.weight_gradients_ewma = np.zeros_like(self.layer.weights) 
        self.bias_gradients_ewma = np.zeros_like(self.layer.biases)

        self.weight_gradients_squared_ewma = np.zeros_like(self.weight_gradients_ewma)
        self.bias_gradients_squared_ewma = np.zeros_like(self.bias_gradients_ewma)
        


    def update(self):
        self.t += 1

        self.weight_gradients_ewma *= self.beta1
        self.weight_gradients_ewma += (1 - self.beta1) * self.layer.weight_gradients

        self.bias_gradients_ewma *= self.beta1
        self.bias_gradients_ewma += (1 - self.beta1) * self.layer.bias_gradients

        self.weight_gradients_squared_ewma *= self.beta2 
        squared_weight_gradients = self.layer.weight_gradients * self.layer.weight_gradients
        squared_weight_gradients *= (1 - self.beta2)
        self.weight_gradients_squared_ewma += squared_weight_gradients

        self.bias_gradients_squared_ewma *= self.beta2 
        squared_bias_gradients = self.layer.bias_gradients * self.layer.bias_gradients
        squared_bias_gradients *= (1 - self.beta2)
        self.bias_gradients_squared_ewma += squared_bias_gradients


        denominator_ewma = 1 - self.beta1 ** self.t 
        denominator_squared_ewma = 1 - self.beta2 ** self.t 


        bias_corrected_weights_ewma = self.weight_gradients_ewma / denominator_ewma
        bias_corrected_bias_ewma = self.bias_gradients_ewma / denominator_ewma

        bias_corrected_weights_squared_ewma = self.weight_gradients_squared_ewma / denominator_squared_ewma
        bias_corrected_bias_squared_ewma = self.bias_gradients_squared_ewma / denominator_squared_ewma


        weight_denominator =  np.sqrt(bias_corrected_weights_squared_ewma) + self.epsilon
        bias_denominator = np.sqrt(bias_corrected_bias_squared_ewma) + self.epsilon


        self.layer.weights -=  bias_corrected_weights_ewma / weight_denominator * self.learning_rate
        self.layer.biases -= bias_corrected_bias_ewma / bias_denominator * self.learning_rate