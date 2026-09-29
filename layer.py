import numpy as np
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
               high = np.sqrt(6 / input_size)
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
        # => ∂before_activation/∂w1 = x1 (w1 is basically the weight)
        #∂Loss/∂weight = input * ∂Loss/∂before_activation
        #we have to transpose because:
        # inputs shape:(batch_size, input_neurons)
        # d_before_activation shape: (batch_size, output_neurons)
        # weights shape: (input_neurons, output_neurons)
        self.weight_gradients = self.inputs.T @ d_before_activation

        #∂Loss/∂b = ∂Loss/∂before_activation * ∂before_activation/∂b
        #∂before_activation/∂b = 1, because: x1*w1 + x2*w2 + b
        #we add up the error of all the examples
        #(we add them up column by column, and we keep the dimension too so it has the same shape as the bias)
        #it's not a problem that only a new pointer is put on the matrix, because neither of them gets modified
        self.bias_gradients = d_before_activation

        #we build the error signal of the previous layer too
        #∂Loss/∂x1 = ∂Loss/∂before_activation * ∂before_activation/∂x1
        #since before_activation = x1*w1 + x2*w2 + b =>
        #=> ∂before_activation/∂x1 = w1
        #the reason for the transposing is similar
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
        self.input_channels = input_channels #what shape of input it's made for, e.g. 1 for grayscale, 3 for RGB
        self.filter_count = filter_count    #how many filters we want (it's usual to use multiple filters in one layer)
        self.filter_size = filter_size
        self.filter_shape = (filter_count, filter_size[0], filter_size[1], input_channels)

        self.activation = activation_function
        self.stride = stride
        self.keep_size = keep_size
        self.filter_initialization_method = filter_initialization_method

        if keep_size:
            if stride != 1:
                raise ValueError("keep_size=True so please select stride = 1")  #we usually only pad with stride 1.
                #in other cases it works out mathematically but it's too expensive. usually with stride > 1 the goal is also to make the output smaller

            if filter_size[0] % 2 == 0 or filter_size[1] % 2 == 0:
                raise ValueError("keep_size=True needs odd filter dimensions")
                #so the edge can fit onto the middle

            self.padding_height = filter_size[0] // 2
            self.padding_width = filter_size[1] // 2
        else:
            self.padding_height = 0
            self.padding_width = 0

        #note, the filter's name stayed weights, because we refer to it as weights in the optimizer
        self.inputs = None
        self.before_activation = None
        self.after_activation = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.optimizer = None
        self.biases = np.zeros(shape=filter_count)
        self.weights = None
        
        #input_size = filter_wifth * filter_height * (how many values belong to the image e.g. 3 for rgb)
        #even though the image is rgb, the filter also grows the same way, since the dot product sum will return one value
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

    #inputs.shape == (batch_size, height, width, channels) this is the shape of the input
    #the point: we cut out a small cube, whose width and height = the width and height of the filter
    #and its 3rd dimension is how many channels there are (e.g. gray scale 1 and rgb 3)
    #then we multiply this with our filter using dot product, add up the values (so we get
    # how well the whole filter matches that part) + bias
    #we do this with every part we can
    def convolve(self, inputs):
        inputs = np.pad(inputs,
            ((0, 0), #the start and end of the batch don't get zeros
             (self.padding_height, self.padding_height), #the height gets zeros at the start and at the end too
             (self.padding_width, self.padding_width), #the width too, like the height
             (0, 0)),  #the start and end of the channel don't get zeros
             mode="constant")   #fill it up with zeros

        #the size of the output is how many steps the filter can take on that axis
        output_height = (inputs.shape[1] - self.filter_size[0]) // self.stride + 1
        output_width = (inputs.shape[2] - self.filter_size[1]) // self.stride + 1

        output = np.zeros(shape=(inputs.shape[0], output_height, output_width, self.filter_count))

        for sample_index in range(inputs.shape[0]): #batches
            for y in range(output_height):
                for x in range(output_width):
                    #where we start from
                    y_start = y * self.stride
                    x_start = x * self.stride

                    #we cut out the right part from the input, that we calculate with the filter
                    input_part = inputs[
                        sample_index,   #which batch
                        y_start : y_start + self.filter_size[0],    #from where to where
                        x_start : x_start + self.filter_size[1],
                        :   #all channels
                    ]
                    
                    #output.shape = (batch_size, output_height, output_width, filter_count) 
                    #and the filter count will be the channel value of the next layer
                    #numpy solves it with broadcasting and handles multiple filters at once
                    #the right part of the output = [which batch, ]
                    output[sample_index, y, x, :] = (   #we get the results of all the filters at the right position
                        np.sum(input_part * self.weights, axis=(1, 2, 3))   #we add up the dot products along height, width and channels
                        + self.biases
                    )

        return output
                
    def forward(self, inputs):
        self.inputs = inputs

        output = self.convolve(inputs)  #the forward is the same as at the dense, only here we have to go through with the filter etc.
        activated = self.activation.f(output)

        self.before_activation = output
        self.after_activation = activated

        return activated
    
    def backward(self, d_after_activation):
        #how much does the one before the activation influence the result
        d_before_activation = d_after_activation * self.activation.df(self.before_activation)

        padded_inputs = np.pad(self.inputs,
            ((0, 0),
             (self.padding_height, self.padding_height),
             (self.padding_width, self.padding_width),
             (0, 0)),
            mode="constant")    #padding like at the forward

        d_padded_inputs = np.zeros_like(padded_inputs)  #we collect the gradients on the input here
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
                    ]   #we filter out from the input what the given output position represents

                    for filter_index in range(self.filter_count):   #one by one for all the filters
                        #self.weight_gradients = self.inputs.T @ d_before_activation
                        #same idea, only here for many small parts that we have to filter out first
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
                        )   #we build up how much the whole padded input influenced the result

        height_slice = slice(None)  #same as ":"
        width_slice = slice(None)

        #note: we use it like this because with slice it's the safest
        # and simple to handle, even when the two paddings aren't 0 or non-0 at the same time
        if self.padding_height != 0:
            height_slice = slice(self.padding_height, -self.padding_height)

        if self.padding_width != 0:
            width_slice = slice(self.padding_width, -self.padding_width)

        d_inputs = d_padded_inputs[:, height_slice, width_slice, :]

        return d_inputs
        

#its role is to turn the 4d data of the conv. layer into 2d (so the dense layer can receive it)
class FlattenLayer:
    def __init__(self):
        self.inputs = None  #we save the input, so we can use it in backprop
        self.weight_gradients = np.array(0)
        self.bias_gradients = np.array(0)   #dummy gradients because of the nn's train
        self.optimizer = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs.reshape(inputs.shape[0], -1)  #batch size stays,
        #the meaning of -1: calculate automatically how big the dimension needs to be here, so all the elements are kept

    def backward(self, d_after_activation):
        return d_after_activation.reshape(self.inputs.shape)    #inverse of the forward, we use the original shape

class MaxPoolLayer:
    def __init__(self, pool_size: tuple[int, int] = (2, 2), stride: int = 2):
        self.pool_size = pool_size  #from how big a "window" we only keep the maximum value
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
        #same logic as in convolve

        for sample_index in range(inputs.shape[0]):
            for y in range(output_height):
                for x in range(output_width):
                    input_part = inputs[
                        sample_index,
                        y * self.stride : y * self.stride + self.pool_size[0],
                        x * self.stride : x * self.stride + self.pool_size[1],
                        :
                    ]   #we filter out the given part
                    output[sample_index, y, x, :] = np.max(input_part, axis=(0, 1)) #we only keep the maximum
                    #(overall maximum = max(row maximum, column maximum), that's why there's the axis)
        return output

    # the dimension grows back but only the maximum from the part covered by every filter can get a gradient,
    # because if we sent that forward, only that could influence the other layers. (the others get 0)
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
                    ]   #from the original we filter out the part whose maximum we got back from the previous layer

                    max_values = np.max(input_part, axis=(0, 1), keepdims=True)
                    mask = input_part == max_values #boolean array mask, based on
                    #where the max value is (because of keepdims we can compare)

                    #in python you can multiply an int with booleans, true = 1, false = 0
                    #where the max value was it lets the gradient through, elsewhere it zeroes it out
                    d_inputs[
                        sample_index,
                        y * self.stride : y * self.stride + self.pool_size[0],
                        x * self.stride : x * self.stride + self.pool_size[1],
                        :
                    ] += mask * d_after_activation[sample_index, y, x, :]

        return d_inputs

