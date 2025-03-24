Objective
Implement a three-layer neural network using only the TensorFlow library (without Keras) to classify MNIST handwritten digits. This implementation demonstrates both feed‑forward propagation and back‑propagation (training) using TensorFlow’s low‑level APIs.

Description of the Model
1.Tensorflow library provides interface to artificial neural network.
2.The MNIST dataset is loaded.
3.Feature engineering is done as normalization.
4.An input layer with 784 neurons (flattened 28x28 images)
5.Two hidden layers with 128 and 64 neurons, using Sigmoid activation function.
6.An output layer with 10 neurons (corresponding to digit classes).
7.Epoch: 20 is used.
8.Epoch: If the model keeps improving, it is advisable to try a higher number of epochs. If the model stopped improving way before the final epoch, it is advisable to try a lower number of epochs.
9.Batch size - 100
10.Batch size refers to the number of samples used in one iteration.
11.Optimization via Adam optimizer to minimize loss.
12.Loss function: Softmax cross entropy is used

Python Implementation
in exp3.ipynb file



Description of Code

1.Load and Preprocess Data:
Loads MNIST dataset (handwritten digits).
Reshapes images (28×28 → 784 pixels).
Normalizes pixel values (0-1 range).
Converts labels to one-hot encoding.

2️.Initialize Model Parameters:
Defines placeholders (X for input, y for labels).
Initializes weights (W1, W2, W3) and biases (b1, b2, b3).

3️.Forward Propagation:
Layer 1: sigmoid(X * W1 + b1).
Layer 2: sigmoid(layer1 * W2 + b2).
Output Layer: layer2 * W3 + b3.

4️.Backpropagation & Optimization:
Uses Softmax Cross-Entropy Loss.
Optimizes weights using Adam Optimizer.

5️.Training the Model:
Uses mini-batch gradient descent.
Prints loss and accuracy at each epoch.

6.Accuracy Metric
argmax(logits, 1): Gets the predicted class.
argmax(y, 1): Gets the actual class.
equal(): Compares predicted vs actual class.
reduce_mean(): Computes the accuracy.

Performance Evaluation
The model achieves high accuracy (~99% on training and ~97% on test data).
The performance is satisfactory for MNIST classification.


My Comments
1. We have used sigmoid activation function in this which is slowing the learning.
 2. We have used disable eager execution function to use tensorflow 1.x(Older Version)

 *Improvements--*
 1. We can use ReLu activation function for speeding up training and increasing model performance.
 2. We can use tensorflow newer version with Keras for easy implementation.
