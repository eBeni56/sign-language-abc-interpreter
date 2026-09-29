from data_reader import get_data
from neural_network import Neural_Network
import activation_function as activation
import loss_function as loss
import layer
import matplotlib.pyplot as plt
import pickle

CSV_FILE_PATH = "data/new_data.csv"

training_inputs, training_outputs, testing_inputs, testing_outputs = get_data(CSV_FILE_PATH)

number_of_classes = training_outputs[0].shape[0] 

my_neural_network = Neural_Network(
    optimizer = "adam",
    layers=[
        layer.DenseLayer(42, 64, activation.ReLU(), "he_normal"),
        layer.DenseLayer(64, 128, activation.ReLU(), "he_normal"),
        layer.DenseLayer(128, 64, activation.ReLU(), "he_normal"),
        layer.DenseLayer(64, 32, activation.ReLU(), "he_normal"),
        layer.DenseLayer(32, number_of_classes, activation.SoftMax(), "xavier_glorot_uniform")
    ],
    loss_function=loss.CategorialCrossEntropy(),
    learning_rate=0.0005
)


epochs = 100
batch_size = 32

train_loss_history, test_loss_history = my_neural_network.train(
        training_inputs= training_inputs, 
        training_outputs= training_outputs, 
        epochs= epochs, 
        batch_size= batch_size, 
        LOG = True, 
        testing_inputs= testing_inputs, 
        testing_outputs= testing_outputs
    )


SAVE_PATH = "models/own_nn_2.pkl"

with open(SAVE_PATH, "wb") as f:
    pickle.dump(my_neural_network, f)


epoch_range = list(range(1, epochs + 1))

plt.plot(epoch_range, train_loss_history)
plt.plot(epoch_range, test_loss_history)
plt.show()