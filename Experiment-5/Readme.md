Objective:
To train and evaluate a convolutional neural network (CNN) using Keras to classify the Fashion MNIST dataset and demonstrate the effects of filter size, regularization, batch size, and optimization algorithm on model performance.

Description of the Model:
The CNN consists of the following layers:
Convolutional Layers: Two convolutional layers with configurable filter sizes (3x3 or 5x5) and ReLU activation.
Pooling Layers: Max pooling layers (2x2) to downsample feature maps.
Fully Connected Layer: A dense layer with 128 neurons and ReLU activation.
Output Layer: A softmax layer with 10 neurons (corresponding to 10 Fashion MNIST classes) for classification.
Hyperparameters varied in experiments:
Filter Size: 3x3 or 5x5
Regularization: L2 regularization (0.0001 or 0.001)
Batch Size: 32 or 64
Optimization Algorithm: Adam or SGD

Python Implementation:
in exp5.ipynb

Performance Evaluation:
Accuracy: The model performance varied based on the hyperparameters. Generally, Adam optimizer outperformed SGD, and larger filter sizes with moderate regularization performed better.
Confusion Matrix: The model made errors in classifying similar clothing items, like distinguishing between a shirt and a t-shirt.
Loss Curve: Adam showed faster convergence compared to SGD, and models with larger filters tended to overfit slightly.

My Comments:
Limitations:
Training time increases significantly with larger filter sizes and higher regularization.
The model does not include advanced data augmentation, which could improve generalization.
Only basic hyperparameter tuning was done; techniques like grid search or Bayesian optimization could yield better results.
The dataset contains visually similar categories, making classification inherently difficult.
Testing on additional datasets would better assess the model’s robustness.
Future improvements include adding dropout layers for better regularization, experimenting with different architectures like ResNet, and using automated hyperparameter tuning methods for optimization.