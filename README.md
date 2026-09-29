# Sign Language ABC - Neural Network from Scratch

A neural network I built from scratch with NumPy (no PyTorch or TensorFlow) that recognizes the sign language alphabet in real time from your webcam.

I made this with a peer as a side project. MediaPipe finds the hand in the webcam picture and gives us 21 landmark points, and our own network takes those points and guesses which letter you're showing.

Only the letters A-Y are in there, without J and Z, because those two need movement and we only look at single frames.

## How it works

1. The webcam frame goes into MediaPipe, which gives back the 21 hand landmarks (x and y each, so 42 numbers)
2. Those 42 numbers go into our neural network
3. The network gives back a probability for each of the 24 letters and we show the best one on the screen

## Main Files

| `layer.py` | The layers (dense, dropout, convolutional, flatten, max pool) with forward and backward pass |
| `activation_function.py` | ReLU, LeakyReLU, Sigmoid, Tanh, SoftPlus, SoftMax |
| `loss_function.py` | Mean squared error, categorical and binary cross entropy |
| `optimizer.py` | Gradient descent, momentum, AdaGrad, RMSProp, Adam |
| `neural_network.py` | Puts it all together: feed forward, backpropagation and the training loop |

## Scripts you can run

| `train_own_model.py` | Trains our own network on the data and saves it to `models/own_nn_2.pkl` |
| `predict_own_model.py` | The webcam demo, shows the predicted letter live |
| `train_tensorflow_model.py` | Same thing but with TensorFlow/Keras, just to compare with ours |
| `data_reader.py` | Reads the csv, one-hot encodes the letters and splits into train/test (used by the training scripts) |

## data folder

| `data/new_data.csv` | The data we train on. One row = one letter + 42 landmark coordinates. Made by us with a script that records the hand from the webcam |
| `data/data.csv` | An older recording made the same way, the training scripts don't use it |

## models fodler

| `models/own_nn_2.pkl` | Our trained network, the demo uses this one |
| `models/own_nn_1.pkl` | An older, smaller version of it |
| `models/hand_landmarker.task` | The MediaPipe hand model from Google |

Only the dense layers are used for the sign language recognition. The convolutional, max pool and dropout layers are in there too, but we didn't need them for this project.

## How to run it

You need Python 3.12 (mediapipe is picky about the Python version)

Install the packages:

```
pip install -r requirements.txt
```

Run everything from the main project folder, because the file paths are relative.

## uses the already trained model:

```
python predict_own_model.py
```

Show a letter to the camera and it writes the prediction in the corner. Press `q` to quit.

## Train the network yourself:

```
python train_own_model.py
```

It trains for 100 epochs, which takes a few minutes, then saves the model to `models/own_nn_2.pkl` (this overwrites the old one!) and shows a plot of the train and test loss.

## Train the TensorFlow version:

```
python train_tensorflow_model.py
```

This saves `tensorflow_model.keras` in the main folder. TensorFlow is a big install and only this script needs it.

## Notes

- The saved `.pkl` models remember the names of our files and classes, so don't rename them or loading the model will break.
- The train/test split is random, so the test results are better than what you get in real life with a different person or lighting. It still reaches around 75% with a new data.
