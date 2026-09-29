from abc import ABC, abstractmethod
import numpy as np

#lusta voltam mindegyikhez szepen kommentet irni magyarazattal, azt majd megoldom holnap
#amugyis atbeszeljuk
#amugy a backward vegul ugyanazt a logikat hasznalta az osszesnel, az updateok csinalnak mast
#ezert azt innen kiszedtem

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

        #ugyanaz shape es dtype mint a suly es bias-gradiensvektoroknak
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



# #Adaptive Gradient, azaz adaptalja a gradiens erosseget, annak fuggvenyeben hogy mennyire
# #sokat "haladott" elore egy adott parameter (mennyire nagy frissitesek voltak rajta) 
# #mig a sima GradientDescent valtozatainal, a learning_rate az csak egyetlen konstans, ami
# #barmilyen parameterre ugyanaz, itt skalazzuk a learning_rate-t annak fuggvenyeben, hogy
# #mennyire nagy gradiensek voltak egy adott parameternel (legyen az suly vagy bias)
# #ezzel megoldodik az hogy pl van egy suly aminek folyton oriasi gradiensei vannak es van egy
# #bias aminek kicsi gradiensei vannak es igy sokkal inkabb a sulynak megfelelo iranyba haladnank, mivel
# #a sulynal tortennek a nagyobb frissitesek, mig az adaptive modszerek skalazzak ilyenkor 
# #a learning_ratet az adott parameternel es sokkal kisebb szorzoja lesz igy ennek
# #ezzel sokkal egyenesebb ut lesz a minimum fele
class AdaGrad(Optimizer):
    def __init__(self, layer, learning_rate, eps = 1e-8):
        super().__init__(layer, learning_rate)

        #epszilon egy kicsi szam, a 0-val valo osztas elkerulesere szolgal 
        self.eps = eps

        # ugyanaz shape es dtype mint a suly es bias-gradiensvektoroknak
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

        #ugyanaz shape es dtype mint a suly es bias-gradiensvektoroknak
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

        #epszilon egy kicsi szam, a 0-val valo osztas elkerulesere szolgal 
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

        #t az idopillanat, azaz konkretan egy szamlalo hogy hanyadjara tortenik frissites
        #erre szukseg van a bias correctionnel 
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