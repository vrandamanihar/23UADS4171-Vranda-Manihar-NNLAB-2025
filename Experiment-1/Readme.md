Experiment: Learning XOR and NAND using Perceptron Learning Algorithm

Objective:
To implement the Perceptron Learning Algorithm using numpy and evaluate its performance on NAND and XOR truth tables.

Model Description:
The perceptron is a simple single-layer neural network with an input layer and an output neuron. It uses a step activation function to determine class labels. The perceptron is trained using the perceptron learning rule, which updates weights iteratively based on classification errors.

Python Implementation:
The code initializes weights randomly, trains the perceptron using the perceptron learning algorithm, and evaluates its performance on the NAND and XOR datasets.

Description of Code:
1. Define a Perceptron class with forward propagation and weight updates.
2. Train the perceptron on NAND and XOR datasets using the perceptron learning rule.
3. Evaluate the performance using accuracy and confusion matrix.
4. Analyze the performance differences between NAND and XOR classification.

Performance Evaluation:
- Accuracy for NAND: {accuracy_nand * 100:.2f}%
- Confusion Matrix for NAND:
{conf_matrix_nand}

- Accuracy for XOR: {accuracy_xor * 100:.2f}%
- Confusion Matrix for XOR:
{conf_matrix_xor}

My Comments:
The perceptron successfully learns the NAND function as it is linearly separable.

However, it fails to learn XOR because XOR is not linearly separable, highlighting the limitation of single-layer perceptrons.

Using a multi-layer perceptron (MLP) with a hidden layer can overcome this limitation and successfully classify XOR.
