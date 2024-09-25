import numpy as np


class ActivationFunctions:

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def softmax(self, inputs):
        # Softmax activation function
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Subtracting max for numerical stability
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def tanh(self, inputs):
        return np.tanh(inputs)


af = ActivationFunctions()

# Example input array
input_array = np.array([1.0, 2.0, -3.0, -4.0])
# Compute sigmoid for all elements
sigmoid_output = af.sigmoid(input_array)
# print(sigmoid_output)

# Compute relu for all elements
relu_output = af.relu(input_array)
# print(relu_output)

# input for softmax
_input = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

result = af.softmax(inputs=_input)
print(result)
