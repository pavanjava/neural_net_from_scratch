import numpy as np
import numpy as np


def neuron_with_3inputs(inputs, weights, bias):
    neural_op = np.dot(inputs, np.transpose(weights)) + bias
    return neural_op


def neuron_with_layer_and_batch(inputs, weights, bias):
    return np.dot(inputs, np.transpose(weights)) + bias


# computing the single neuron
x = np.array([1, 2, 3])
y = np.array([0.2, 0.8, -0.5])
bias = 2
print(neuron_with_3inputs(x, y, bias))

# computing the neuron as layer
x = np.array([1.0, 2.0, 3.0, 2.5])
y = np.array([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.32, -0.56], [-0.26, -0.45, 0.17, 0.87]])
bias = [2.0, 3.0, 0.5]
print(neuron_with_layer_and_batch(x, y, bias))

# computing the neuron as layer with batch input
x = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.5]])
y = np.array([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.32, -0.56], [-0.26, -0.45, 0.17, 0.87]])
bias = np.array([[2.0, 3.0, 0.5], [2.0, 3.0, 0.5], [2.0, 3.0, 0.5]])
print(neuron_with_layer_and_batch(x, y, bias))
