Experiment: Learning XOR and NAND using Perceptron Learning Algorithm

Objective:
To implement the Perceptron Learning Algorithm using numpy and evaluate its performance on NAND and XOR truth tables.

Model Description:
The perceptron is a simple single-layer neural network with an input layer and an output neuron. It uses a step activation function to determine class labels. The perceptron is trained using the perceptron learning rule, which updates weights iteratively based on classification errors.

Python Implementation:
(in ex1.ipynb file)

Description of the Code
Perceptron Class:

Initialization: Sets weights and bias to 0.
Activation Function: Classifies inputs (outputs 1 if the weighted sum is positive; otherwise, 0).
Predict Function: Uses the activation function to output predictions.
Train Function: Adjusts weights to learn from the input data.
Evaluate Function: Computes the accuracy of the predictions.
Training and Evaluation for NAND Gate:

Data: Defines the NAND truth table and corresponding labels.
Training: The model is trained using the train() method.
Evaluation: Accuracy is calculated using the evaluate() function.
Results:
Accuracy: 100%
Predictions:
[0, 0] → 1
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
Training and Evaluation for XOR Gate:

Data: Defines the XOR truth table and labels.
Training: The perceptron is trained on the XOR data.
Evaluation: Accuracy is computed and predictions printed.
Results:
Accuracy: 50%
Predictions:
[0, 0] → 1
[0, 1] → 1
[1, 0] → 0
[1, 1] → 0


Performance:
NAND Gate
- Accuracy: 1.0 (100%)
- Predictions: Correct for all inputs.

XOR Gate
- Accuracy: Approximately 0.5 (50%)
- Predictions: Incorrect for half inputs.


My comments:
	The perceptron cannot work properly on non-linearly separable data. Because of this, the perceptron is unable to learn the XOR gate.
	To improve this, we can use multi-layer perceptron.
