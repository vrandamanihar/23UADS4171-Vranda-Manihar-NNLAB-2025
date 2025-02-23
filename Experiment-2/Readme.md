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
Apperceptron(inputs, weights, bias):

Computes 
net
=
inputs
⋅
weights
+
bias
net=inputs⋅weights+bias.
Returns 1 if 
net
>
0
net>0, else 0.
train_perceptron(X, y):

Iterates over all training samples multiple times (epochs).
Updates weights and bias using the error 
(
target
−
output
)
(target−output).
Hidden Perceptrons Training:

Each hidden perceptron is trained to replicate the desired logical function (e.g., AND-like).
After training, each neuron’s weights/bias are fixed.
Final XOR Perceptron:

Takes the outputs from the hidden layer as inputs.
Trained to produce XOR = 1 for 
[0,1]
[0,1] and 
[1,0]
[1,0] and 0 otherwise.
Execution:

Once trained, the network is tested on the four possible XOR inputs.
The outputs are printed.


Performance Evaluation
Output:
Input: [0 0] -> XOR Output: 0
Input: [0 1] -> XOR Output: 1
Input: [1 0] -> XOR Output: 1
Input: [1 1] -> XOR Output: 0

Results:
The MLP successfully computes the correct XOR outputs for all input combinations.



My Comments
Current Approach:
The hidden perceptrons are each trained separately to mimic specific logical operations. While this demonstrates the concept, it isn’t fully scalable to more complex tasks.

Future Improvement:
Implementing a proper backpropagation algorithm would allow end-to-end training of the entire network and better generalization for more complex problems.