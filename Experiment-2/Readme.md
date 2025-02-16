Objective
Implement a Multi-Layer Perceptron (MLP) with one hidden layer using NumPy to learn the XOR Boolean function.

Description of the Model
Input Layer: Takes XOR inputs.
Hidden Layer: Contains 4 neurons with preset weights/biases for basic logic functions (e.g., AND, OR, NOR).
Output Layer: Combines selected hidden neuron outputs to produce the final XOR result.

Python Implementation
(in exp2.ipynb file)


Description of the Code
Perceptron Function:
Applies a weighted sum and bias to generate binary outputs.

XOR MLP Function:

Defines weights/biases for 4 hidden neurons.
Computes hidden outputs.
Combines these outputs and calculates the final XOR output.
Execution:
Processes the XOR truth table inputs and prints the corresponding outputs.

Performance Evaluation
Output:
Input: [0 0] -> XOR Output: 0
Input: [0 1] -> XOR Output: 1
Input: [1 0] -> XOR Output: 1
Input: [1 1] -> XOR Output: 0

Results:
The MLP successfully computes the correct XOR outputs for all input combinations.



My Comments
1.The weights and biases are learnt previously therefore can't be scalable for more complex tasks.
2.Implementing a training phase (e.g., backpropagation) would allow the network to learn from data automatically and adapt to more complex problems.