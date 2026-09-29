import numpy as np
import optimizer as op
from activation_function import ActivationFunction

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation_function: ActivationFunction, weight_initialization_method: str = None):
        
        self.input_size = input_size
        self.output_size = output_size

        self.biases = np.zeros(shape = (1, output_size))

        self.activation = activation_function
        self.inputs = None
        self.before_activation = None
        self.after_activation = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.optimizer = None


        match weight_initialization_method:
           case "he_normal":
               stdev = np.sqrt(2 / input_size)
               self.weights = np.random.normal(loc = 0, scale = stdev,
                       size = (input_size, output_size))

           case "he_uniform":
               low = -np.sqrt(6 / input_size)
               high = np.sqrt(6 / output_size)
               self.weights = np.random.uniform(low = low, high = high,
                       size = (input_size, output_size))

           case "xavier_glorot_normal":
               stdev = np.sqrt(2 / (input_size + output_size))
               self.weights = np.random.normal(loc = 0, scale = stdev,
                       size = (input_size, output_size))

           case "xavier_glorot_uniform":
               value = np.sqrt(6 / (input_size + output_size))
               self.weights = np.random.uniform(low = -value, high = value,
                       size = (input_size, output_size))
            
           case _:                    
               self.weights = np.random.randn(input_size, output_size)
    

    def initialize_optimizer(self, optimizer_from_nn):
        self.optimizer = optimizer_from_nn

    def forward(self, inputs):
        self.inputs = inputs

        output = inputs @ self.weights + self.biases
        activated = self.activation.f(output)
        
        self.before_activation = output
        self.after_activation = activated

        return activated    
    

    def backward(self, d_after_activation):
        # ∂Loss/∂before_activation = ∂Loss/∂after_activation * ∂after_activation/∂before_activation
        d_before_activation = d_after_activation * self.activation.df(self.before_activation)

        #∂Loss/∂weight = ∂Loss/∂before_activation * ∂before_activation/∂weight
        #before_activation = x1*w1 + x2*w2 + b =>  
        # => ∂before_activation/∂w1 = x1 (w1 alapbol a weight)
        #∂Loss/∂weight = input * ∂Loss/∂before_activation
        #transzponalni kell mivel: 
        # inputs shape:(batch_size, input_neurons)
        # d_before_activation shape: (batch_size, output_neurons)
        # weights shape: (input_neurons, output_neurons)
        self.weight_gradients = self.inputs.T @ d_before_activation

        #∂Loss/∂b = ∂Loss/∂before_activation * ∂before_activation/∂b
        #∂before_activation/∂b = 1, mivel: x1*w1 + x2*w2 + b 
        #osszeadjuk az osszes peldara vonatkozo hibat 
        #(oszloponkent adjuk ossze, es megtartjuk a dimenziot is hogy ugyan olyan formaju legyen mint a bias)
        #nem zavar be, hogy csak egy uj mutato kerul a matrixra, mert nem modosul egyik sem
        self.bias_gradients = d_before_activation

        #felepitjuk az elozo layer hiba jelet is
        #∂Loss/∂x1 = ∂Loss/∂before_activation * ∂before_activation/∂x1
        #mivel before_activation = x1*w1 + x2*w2 + b =>
        #=> ∂before_activation/∂x1 = w1
        #transponalas oka hasonlo
        d_inputs = d_before_activation @ self.weights.T

        return d_inputs

class DropoutLayer:
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate  
        self.scaler = 1.0 - self.dropout_rate
        self.active = True

    def forward(self, inputs):
        if self.active:
            self.mask = np.random.rand(*inputs.shape) > self.dropout_rate
            return np.where(self.mask, inputs / self.scaler, 0.0)
        
        return inputs
    

    def backward(self, d_after_activation):
        if self.active:
            return np.where(self.mask, d_after_activation / self.scaler, 0.0)
        
        return d_after_activation


class ConvolutionalLayer:
    def __init__(self, input_channels: int, filter_count: int, filter_size: tuple[int, int], stride: int, activation_function: ActivationFunction, keep_size: bool = True, filter_initialization_method: str = None):
        self.input_channels = input_channels #milyen alaku inputra keszuljon pl grayscale eseten 1, RGB eseten 3
        self.filter_count = filter_count    #hany filtert akarunk (egy layerben szokas tobb filtert hasznalni)
        self.filter_size = filter_size
        self.filter_shape = (filter_count, filter_size[0], filter_size[1], input_channels)

        self.activation = activation_function
        self.stride = stride
        self.keep_size = keep_size
        self.filter_initialization_method = filter_initialization_method

        if keep_size:
            if stride != 1:
                raise ValueError("keep_size=True so please select stride = 1")  #csak stride 1 eseten szoktak paddingelni. 
                #mas esetben matematikailag kijon de tul koltseges. altalaban stride > 1 eseten az is a cel, hogy kisebb legyen az output

            if filter_size[0] % 2 == 0 or filter_size[1] % 2 == 0:
                raise ValueError("keep_size=True needs odd filter dimensions")
                #hogy a szele ra tudjon illeszkedni a kozepere

            self.padding_height = filter_size[0] // 2
            self.padding_width = filter_size[1] // 2
        else:
            self.padding_height = 0
            self.padding_width = 0

        #megjegyzes, a filter neve maradt weights, mert az optimezerben weights-kent hivatkozunk ra
        self.inputs = None
        self.before_activation = None
        self.after_activation = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.optimizer = None
        self.biases = np.zeros(shape=filter_count)
        self.weights = None
        
        #input_size = filter_wifth * filter_height * (hany ertek tartozik a kephez pl rgbnel = 3)
        #hiaba rgb a kep, a filter is csak ugyan ugy no, mivel dot product sumb-ol fog erteket vissza adni
        #output size = filter_wifth * filter_height * filter_count
        match filter_initialization_method:
           case "he_normal":
               stdev = np.sqrt(2 / (self.filter_size[0] * self.filter_size[1] * self.input_channels))
               self.weights = np.random.normal(loc = 0, scale = stdev,
                       size = self.filter_shape)

           case "he_uniform":
               limit = np.sqrt(6 / (self.filter_size[0] * self.filter_size[1] * self.input_channels))
               self.weights = np.random.uniform(low = -limit, high = limit,
                       size = self.filter_shape)

           case "xavier_glorot_normal":
               stdev = np.sqrt(2 / (self.filter_size[0] * self.filter_size[1] * self.input_channels + self.filter_size[0] * self.filter_size[1] * self.filter_count))
               self.weights = np.random.normal(loc = 0, scale = stdev,
                       size = self.filter_shape)

           case "xavier_glorot_uniform":
               value = np.sqrt(6 / (self.filter_size[0] * self.filter_size[1] * self.input_channels + self.filter_size[0] * self.filter_size[1] * self.filter_count))
               self.weights = np.random.uniform(low = -value, high = value,
                       size = self.filter_shape)
            
           case _:                    
               self.weights = np.random.randn(self.filter_count, self.filter_size[0], self.filter_size[1], self.input_channels)
               

    def initialize_optimizer(self, optimizer_from_nn):
        self.optimizer = optimizer_from_nn

    #inputs.shape == (batch_size, height, width, channels) ez az input alakja
    #lenyeg: kivagunk egy kicsi kockat, melynek szelessege es magassaga = a filter szelessege es magassagaval
    #es a 3. dimenzioja pedig h hany chanel van (pl gray scale 1 es rgb 3)
    #majd ezt dot producttal ossze szorozzuk a filterunkkel, ossze adjuk az ertekeket (hogy megkapjuk
    # a teljes filter mennyire illeszkedik az adott reszre) + bias
    #ezt elvegezzuk minden resszel amivel lehet
    def convolve(self, inputs):
        inputs = np.pad(inputs,
            ((0, 0), #a batch eleje es vege nem kap nullat
             (self.padding_height, self.padding_height), #a magassag kap nullat az elejere es a vegere is
             (self.padding_width, self.padding_width), #a szelesseg is mint magassag
             (0, 0)),  #a channel eleje es vege nem kap nullas
             mode="constant")   #zerosokkal toltse fel

        #az output meretei az hogy hany lepest tud megtenni az adott tengelyen a filter
        output_height = (inputs.shape[1] - self.filter_size[0]) // self.stride + 1
        output_width = (inputs.shape[2] - self.filter_size[1]) // self.stride + 1

        output = np.zeros(shape=(inputs.shape[0], output_height, output_width, self.filter_count))

        for sample_index in range(inputs.shape[0]): #batchek (de amugy ezert kerdeztem, 
            #hogy nem e lenne erdemesebb atirni az egeszet ugy, hogy -1 dimenzio, 
            #mert pl itt sem hasznaljuk ki mert ez is mindig 1et kap mint batch)
            for y in range(output_height):
                for x in range(output_width):
                    #honnan kezdunk
                    y_start = y * self.stride
                    x_start = x * self.stride

                    #kivagjuk az inputbol a megfelelo reszt, amire szamolunk a filterrel
                    input_part = inputs[
                        sample_index,   #hanyadik batch
                        y_start : y_start + self.filter_size[0],    #honnan hova
                        x_start : x_start + self.filter_size[1],
                        :   #osszes channel
                    ]
                    
                    #output.shape = (batch_size, output_height, output_width, filter_count) 
                    #es a filter count lesz a kovetkezo layer chanel erteke
                    #numpy broadcastinggal megoldja es lekezzeli egyszerre a tobb filtert
                    #az output megfelelo resze = [hanyadik batch, ]
                    output[sample_index, y, x, :] = (   #a megfelelo poziciora lekerjuk az osszes filter eredmenyet
                        np.sum(input_part * self.weights, axis=(1, 2, 3))   #osszeadjuk a dot productokat magassag, szelesseg es csatornak menten
                        + self.biases
                    )

        return output
                
    def forward(self, inputs):
        self.inputs = inputs

        output = self.convolve(inputs)  #a forward ugyan az mint a densenel, csak itt a filterrel kell vegig jarni stb.
        activated = self.activation.f(output)

        self.before_activation = output
        self.after_activation = activated

        return activated
    
    def backward(self, d_after_activation):
        #mennyire befolyasolja az activacio elotti az eredmenyt
        d_before_activation = d_after_activation * self.activation.df(self.before_activation)

        padded_inputs = np.pad(self.inputs,
            ((0, 0),
             (self.padding_height, self.padding_height),
             (self.padding_width, self.padding_width),
             (0, 0)),
            mode="constant")    #padding mint a forwardnal

        d_padded_inputs = np.zeros_like(padded_inputs)  #ide gyujtjuk az inputra valo gradienseket
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)

        for sample_index in range(self.inputs.shape[0]):
            for y in range(d_before_activation.shape[1]):
                for x in range(d_before_activation.shape[2]):
                    y_start = y * self.stride
                    x_start = x * self.stride

                    input_part = padded_inputs[
                        sample_index,
                        y_start : y_start + self.filter_size[0],
                        x_start : x_start + self.filter_size[1],
                        :
                    ]   #kiszurjuk az inputbol azt amit kepvisel az adott output pozicio

                    for filter_index in range(self.filter_count):   #rendre az osszes filterre
                        #self.weight_gradients = self.inputs.T @ d_before_activation
                        #ugyan ez az otlet, csak itt sok kicsi reszre amit elobb ki kell szurni
                        self.weight_gradients[filter_index] += (input_part * d_before_activation[sample_index, y, x, filter_index])
                        self.bias_gradients[filter_index] += (d_before_activation[sample_index, y, x, filter_index])

                        d_padded_inputs[
                            sample_index,
                            y_start : y_start + self.filter_size[0],
                            x_start : x_start + self.filter_size[1],
                            :
                        ] += (
                            self.weights[filter_index]
                            * d_before_activation[sample_index, y, x, filter_index]
                        )   #felepitjuk a teljes padded input mennyire befolyasolta az eredmenyt

        height_slice = slice(None)  #megegyezik a ":"-el
        width_slice = slice(None)

        #megj: azert hasznaljuk igy, mert slice-al a legbiztonsagosabb, 
        # es egyszeru kezelni, azt is ha nem ugyanakkor 0 vagy nem 0 a ket padding
        if self.padding_height != 0:
            height_slice = slice(self.padding_height, -self.padding_height)

        if self.padding_width != 0:
            width_slice = slice(self.padding_width, -self.padding_width)

        d_inputs = d_padded_inputs[:, height_slice, width_slice, :]

        return d_inputs
        

#szerepe a conv. layer 4d adatat 2d-be alakitani (hogy meg tudja kapni a dense layer)
class FlattenLayer:
    def __init__(self):
        self.inputs = None  #elmentjuk az inputot, hogy majd tudjuk hasznalni backpropnal
        self.weight_gradients = np.array(0)
        self.bias_gradients = np.array(0)   #dummy gradients az nn train-je miatt
        self.optimizer = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs.reshape(inputs.shape[0], -1)  #batch size marad, 
        #1 jelentese: szamoldd ki automatikusan ide mekkora dimenzio kell, hogy az osszes elem megmaradjon

    def backward(self, d_after_activation):
        return d_after_activation.reshape(self.inputs.shape)    #forward inverze, felhasznaljuk az eredeti shapet

class MaxPoolLayer:
    def __init__(self, pool_size: tuple[int, int] = (2, 2), stride: int = 2):
        self.pool_size = pool_size  #mekkora "ablakbol" tartjuk meg csak a maximum erteket
        self.stride = stride
        self.inputs = None
        self.weight_gradients = np.array(0)
        self.bias_gradients = np.array(0)
        self.optimizer = None

    def forward(self, inputs):
        self.inputs = inputs

        output_height = (inputs.shape[1] - self.pool_size[0]) // self.stride + 1
        output_width = (inputs.shape[2] - self.pool_size[1]) // self.stride + 1

        output = np.zeros(shape=(inputs.shape[0], output_height, output_width, inputs.shape[3]))    
        #ugyan az a logika mint a convolve-ban

        for sample_index in range(inputs.shape[0]):
            for y in range(output_height):
                for x in range(output_width):
                    input_part = inputs[
                        sample_index,
                        y * self.stride : y * self.stride + self.pool_size[0],
                        x * self.stride : x * self.stride + self.pool_size[1],
                        :
                    ]   #kiszurjuk az adott reszt
                    output[sample_index, y, x, :] = np.max(input_part, axis=(0, 1)) #megtartjuk csak a maximumot
                    #(teljes maximum = max(sor maximum, oszlop maximum), azert van az axis)
        return output

    # a dimension vissza no de csak minden filter altal lefedett reszbol a maximum kaphat gradienst, 
    # mivel ha azt kuldtuk tovabb, csak az befolyasolhatta a tobbi layert. (a tobbi 0-at kap)
    def backward(self, d_after_activation):
        d_inputs = np.zeros_like(self.inputs)

        for sample_index in range(self.inputs.shape[0]):
            for y in range(d_after_activation.shape[1]):
                for x in range(d_after_activation.shape[2]):
                    input_part = self.inputs[
                        sample_index,
                        y * self.stride : y * self.stride + self.pool_size[0],
                        x * self.stride : x * self.stride + self.pool_size[1],
                        :
                    ]   #az eredetibol kiszurjuk azt a reszt, aminek a maximumat kaptuk vissza az elozo layertol

                    max_values = np.max(input_part, axis=(0, 1), keepdims=True)
                    mask = input_part == max_values #boolean tomb maszk, az alapjan, 
                    #hogy hol van a max ertek (a keepdims miatt tudjuk osszehasonlitani)

                    #pthonban lehet booleanokkal szorozni int-et, true = 1, false = 0
                    #ahol a volt a max ertek ott atengedi a gradienst, mashol lenullazza
                    d_inputs[
                        sample_index,
                        y * self.stride : y * self.stride + self.pool_size[0],
                        x * self.stride : x * self.stride + self.pool_size[1],
                        :
                    ] += mask * d_after_activation[sample_index, y, x, :]

        return d_inputs

