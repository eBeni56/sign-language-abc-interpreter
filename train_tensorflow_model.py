from data_reader import get_data
import tensorflow as tf
import matplotlib.pyplot as plt

CSV_FILE_PATH = "data/new_data.csv"

training_inputs, training_outputs, testing_inputs, testing_outputs = get_data(CSV_FILE_PATH)

number_of_classes = training_outputs.shape[1]

my_neural_network = tf.keras.Sequential([
    tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(),
        input_shape=(42,)
    ),
    tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    )
])

my_neural_network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy()
)

epochs = 100
batch_size = 32

history = my_neural_network.fit(
    training_inputs,
    training_outputs,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_data=(testing_inputs, testing_outputs)
)

train_loss_history = history.history["loss"]
test_loss_history = history.history["val_loss"]

epoch_range = list(range(1, epochs + 1))


my_neural_network.save("tensorflow_model.keras")



plt.plot(epoch_range, train_loss_history)
plt.plot(epoch_range, test_loss_history)
plt.show()