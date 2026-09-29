import numpy as np
import optimizer as optimizer_module
from loss_function import LossFunction, CategorialCrossEntropy
from activation_function import SoftMax
import layer as layer_module
import time 

class Neural_Network:
    def __init__(self, layers: list, loss_function: LossFunction, optimizer: str = "gradient_descent", learning_rate = 0.001):
        
        # warning that SoftMax only works together with CategorialCrossEntropy, so it doesn't fail silently in the background
        if type(layers[-1].activation) == SoftMax and type(loss_function) != CategorialCrossEntropy:
            raise ValueError("SoftMax can only be used with CategorialCrossEntropy") 


        self.layers = layers 
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        self.initialize_optimizers()

    def initialize_optimizers(self):
        for layer in self.layers:
            if isinstance(layer, (layer_module.MaxPoolLayer, layer_module.FlattenLayer, layer_module.DropoutLayer)):
                layer.optimizer = None
                continue

            match self.optimizer:
                case "gradient_descent":
                    layer.initialize_optimizer(optimizer_module.GradientDescent(layer, self.learning_rate))

                case "gradient_descent_momentum":
                    layer.initialize_optimizer(optimizer_module.GradientDescentWithMomentum(layer, self.learning_rate))

                case "ada_grad":
                    layer.initialize_optimizer(optimizer_module.AdaGrad(layer, self.learning_rate))

                case "rms_prop":
                    layer.initialize_optimizer(optimizer_module.RMSProp(layer, self.learning_rate))

                case "adam":
                    layer.initialize_optimizer(optimizer_module.Adam(layer, self.learning_rate))

                case _:
                    raise ValueError("Unknown optimizer")


    def feed_forward(self, inputs, dropout_is_active = True):
        output = inputs

        for layer in self.layers:
            if type(layer) == layer_module.DropoutLayer:
                layer.active = dropout_is_active

            output = layer.forward(output)

        return output
    
    def back_propagation(self, predictions, targets):
        output_layer = self.layers[-1]
        
        if type(output_layer.activation) == SoftMax and type(self.loss_function) == CategorialCrossEntropy:
            # the derivative of softmax is a Jacobian matrix, but when it's paired with categorial cross entropy
            # it simplifies a lot, so we calculate the d_before_activation right here
            # ∂Loss/∂z = ∂Loss/∂a * ∂a/∂z so the derivative of the loss function * the derivative of the activation function
            # the result of this for the softmax + categorial cross entropy combination is: prediction - target
            gradient = predictions - targets  

            # this is exactly the same as in the layer.backward method, calculating the weight gradients and bias gradients
            output_layer.weight_gradients = output_layer.inputs.T @ gradient
            output_layer.bias_gradients = gradient

            # this is basically the d_inputs from layer.backward
            gradient = gradient @ output_layer.weights.T
        

            # and now let's continue from the second to last layer, the last one is already done
            layers = self.layers[:-1]
        else:
            # otherwise just like before, we only calculate the derivative of the loss function first
            # and we go through the last layer as well with the for loop
            gradient = self.loss_function.df(predictions, targets)
            layers = self.layers


        for layer in reversed(layers):
            gradient = layer.backward(gradient)

    
    def calculate_loss(self, predictions, targets):
        return self.loss_function.f(predictions, targets)
    
    def update_parameters(self):
        for layer in self.layers:
            if layer.optimizer is not None:
                layer.optimizer.update()
    

    # the testing dataset can also be given, so the train/val loss is visible too, this way we can see if overfitting happens
    def train(self, training_inputs, training_outputs, epochs, batch_size=1, LOG=True, testing_inputs=None, testing_outputs=None): 
        training_sample_count = training_inputs.shape[0]

        train_loss_history = [0] * epochs
        test_loss_history = None 

        if testing_inputs is not None and testing_outputs is not None:
            test_loss_history = [0] * epochs 


        for epoch in range(epochs):
            start = time.perf_counter()
            
            # shuffle the data, so it's not always the same data in a given batch
            indices = np.random.permutation(training_inputs.shape[0])
            training_inputs = training_inputs[indices]
            training_outputs = training_outputs[indices]

            epoch_loss = 0
            batch_count = 0

            for batch_start_index in range(0, training_sample_count, batch_size):
                batch_end_index = min(batch_start_index + batch_size, training_sample_count)

                # we collect all the gradients of the batch here
                accumulated_weight_gradients = []
                accumulated_bias_gradients = []

                batch_loss = 0
                current_batch_size = batch_end_index - batch_start_index

                for sample_index in range(batch_start_index, batch_end_index):
                    #this way we keep the sample in 2D, we don't just return a plain vector.
                    #in reality it gives a view of the array so no copy is made. that's why it's more efficient in practice
                    #than just wrapping the right index in an np.array()
                    training_input_sample = training_inputs[sample_index:sample_index+1]
                    expected_output_sample = training_outputs[sample_index:sample_index+1]

                    prediction = self.feed_forward(training_input_sample)
                    sample_loss = self.calculate_loss(prediction, expected_output_sample)
                      
                    batch_loss += sample_loss

                    self.back_propagation(prediction, expected_output_sample)

                    #if this is the first sample in the batch, we first set the right sizes
                    if len(accumulated_weight_gradients) == 0:
                        for layer in self.layers:
                            if type(layer) == layer_module.DropoutLayer:
                                continue

                            #zeros_like attaches a matrix of the same size filled with 0s
                            accumulated_weight_gradients.append(np.zeros_like(layer.weight_gradients))
                            accumulated_bias_gradients.append(np.zeros_like(layer.bias_gradients))

                    dropout_layers_till_now = 0 

                    #we add them one by one
                    for layer_index, layer in enumerate(self.layers):
                        if type(layer) == layer_module.DropoutLayer:
                            dropout_layers_till_now += 1
                            continue
                        
                        idx = layer_index - dropout_layers_till_now
                        
                        accumulated_weight_gradients[idx] += layer.weight_gradients
                        accumulated_bias_gradients[idx] += layer.bias_gradients


                dropout_layers_till_now = 0 

                #averaging at the end of the batch
                for layer_index, layer in enumerate(self.layers):
                    if type(layer) == layer_module.DropoutLayer:
                        dropout_layers_till_now += 1
                        continue
                        
                    idx = layer_index - dropout_layers_till_now

                    layer.weight_gradients = accumulated_weight_gradients[idx] / current_batch_size
                    layer.bias_gradients = accumulated_bias_gradients[idx] / current_batch_size
                
                #we only update at the end of the batch
                self.update_parameters()

                batch_loss /= current_batch_size
                epoch_loss += batch_loss
                batch_count += 1

            epoch_loss /= batch_count
            train_loss_history[epoch] = epoch_loss
            
            end = time.perf_counter()

            if LOG:
                print(f"Epoch {epoch}")
                print(f" Train Loss: {epoch_loss}")
                print(f" Duration: {end - start}s")

            if testing_inputs is not None and testing_outputs is not None:
                val_loss = 0

                test_size = testing_inputs.shape[0]
      
                for test_idx in range(test_size):
                    test_input = testing_inputs[test_idx:test_idx+1]
                    test_output = testing_outputs[test_idx:test_idx+1]

                    prediction = self.feed_forward(test_input, False)
                    val_loss += self.calculate_loss(prediction, test_output)
                
                val_loss /= test_size
                test_loss_history[epoch] = val_loss 

                if LOG:
                    print(f" Validation Loss: {val_loss}")

            


        return train_loss_history, test_loss_history