import numpy as np


class ActivationFunctions:

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(inputs))


af = ActivationFunctions()

# Example input array
input_array = np.array([1.0, 2.0, -3.0, -4.0])
# Compute sigmoid for all elements
sigmoid_output = af.sigmoid(input_array)
print(sigmoid_output)

# Compute relu for all elements
relu_output = af.relu(input_array)
print(relu_output)