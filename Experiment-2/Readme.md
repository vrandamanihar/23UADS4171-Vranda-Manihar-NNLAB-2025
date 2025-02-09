**Report: Multi-Layer Perceptron for XOR Function**

**Objective:**
The objective of this experiment is to implement a Multi-Layer Perceptron (MLP) using NumPy to learn the XOR Boolean function. The model is trained using backpropagation and gradient descent to minimize the error.

**Description of the Model:**
This MLP consists of:
- An input layer with 2 neurons corresponding to the XOR inputs.
- A hidden layer with 2 neurons to introduce non-linearity.
- An output layer with 1 neuron producing the final classification result.
- Sigmoid activation function is used in both hidden and output layers.
- Weights and biases are randomly initialized and updated using gradient descent.

**Python Implementation:**
The implementation includes:
- Initialization of weights and biases.
- Forward propagation to compute activations.
- Backpropagation to compute errors and update weights.
- Training for 10,000 epochs with a learning rate of 0.5.
- Performance evaluation using accuracy and confusion matrix.
- Visualization of the loss curve.

**Description of the Code:**
1. **Data Preparation:** The XOR inputs and corresponding outputs are defined.
2. **Network Initialization:** Random weights and biases are assigned.
3. **Training Phase:**
   - Forward propagation computes hidden and output layer activations.
   - Error is computed as the difference between predicted and actual outputs.
   - Backpropagation updates weights using gradient descent.
4. **Performance Evaluation:**
   - The trained model predicts XOR outputs.
   - Accuracy and confusion matrix are calculated.
   - Loss curve is plotted to show training progress.

**Performance Evaluation:**
- **Accuracy:** The model achieves 100% accuracy on the XOR dataset.
- **Confusion Matrix:** A perfect confusion matrix is obtained indicating correct classification of all inputs.
- **Loss Curve:** The loss function decreases over epochs, showing effective learning.

**My Comments:**
- The model effectively learns the XOR function with simple architecture.
- Sigmoid activation can lead to vanishing gradient issues; using ReLU might improve training.
- More hidden neurons or layers may enhance generalization for complex datasets.
- The model is limited to simple binary classification; extending it to multi-class problems would require modifications.

