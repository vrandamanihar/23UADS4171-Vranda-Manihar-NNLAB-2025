Objective
Implement a three-layer neural network using only the TensorFlow library (without Keras) to classify MNIST handwritten digits. This implementation demonstrates both feed‑forward propagation and back‑propagation (training) using TensorFlow’s low‑level APIs.

Description of the Model
Input Layer:

Accepts MNIST images, each flattened into a 784‑dimensional vector (28×28 pixels).
Hidden Layer:

Consists of 256 neurons.
Uses a ReLU activation function to introduce non-linearity.
Output Layer:

Has 10 neurons (one for each digit, 0–9).
Outputs raw logits, which are converted to probabilities using the softmax function.
Training Approach:

Loss Function: Cross‑entropy loss between predicted probabilities and true one‑hot labels.
Optimizer: Adam optimizer is used to perform back‑propagation and update the weights and biases.
Evaluation: The model prints training loss and accuracy for each epoch and reports final test accuracy.

Python Implementation
in exp3.ipynb file



Description of Code
Data Loading and Preparation:

Loading: The MNIST dataset is loaded using tf.keras.datasets.mnist.load_data().
Preprocessing: Images are reshaped from 28×28 to 784-dimensional vectors and normalized. Labels are converted to one-hot encoding.
Placeholders:

x_ph and y_ph are defined to feed input data (images) and true labels into the network.
Network Architecture:

Hidden Layer:
Contains 256 neurons.
Uses a weight matrix W1 and bias b1, with ReLU activation applied to the linear combination.
Output Layer:
Contains 10 neurons for the 10 classes.
Uses a weight matrix W2 and bias b2 to produce logits.
Feed‑Forward Process:

Hidden Layer Computation:
z1 = tf.matmul(x_ph, W1) + b1 computes the linear combination.
a1 = tf.nn.relu(z1) applies ReLU activation.
Output Layer Computation:
logits = tf.matmul(a1, W2) + b2 produces raw outputs.
predictions = tf.nn.softmax(logits) converts logits into probabilities.
Loss and Back-Propagation:

The loss is calculated using cross‑entropy between the predicted probabilities and the true labels.
The Adam optimizer automatically computes gradients and updates the weights and biases during training.
Accuracy Metric:

The model's accuracy is computed by comparing the predicted class (via argmax) with the true class.
Training Loop:

The network is trained for 10 epochs using mini‑batches of 100 samples.
At the end of each epoch, the training loss and accuracy are printed.
After training, the model's test accuracy is evaluated.
Performance Evaluation
Training Performance:

During each epoch, the script prints the current training loss and accuracy.
Test Performance:

At the end of training, the model is evaluated on the test dataset, and the test accuracy is printed (typically around 97%–98% for a simple network on MNIST).
Additional Metrics:

While this script prints accuracy and loss, further evaluation (e.g., confusion matrix or loss curves) can be added with additional code if needed.

My Comments
Limitations:
TF1-Style Code: The implementation uses TensorFlow 2.x in TF1 compatibility mode with placeholders and sessions. Although educational, modern TensorFlow code often uses Keras or eager execution for simplicity.
Basic Model: The network is relatively simple. For more complex tasks, improvements like additional hidden layers, dropout, batch normalization, and more extensive hyperparameter tuning might be necessary.
Scope for Improvement:
Full Backpropagation with Keras: Consider migrating to Keras for a higher-level API that simplifies model building and training.
Advanced Evaluation: Implementing a confusion matrix and plotting loss/accuracy curves would provide deeper insights into model performance.
Regularization: Techniques such as dropout or L2 regularization can help prevent overfitting, especially on more complex datasets.