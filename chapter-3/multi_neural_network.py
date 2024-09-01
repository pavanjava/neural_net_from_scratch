import numpy as np


class MultiLayerNeuralNetwork:
    def __init__(self, inputs: np.array, weights: np.array, bias: np.array):
        self.inputs: np.array = inputs
        self.weights: np.array = weights
        self.bias: np.array = bias

    def compute(self) -> np.array:
        return np.dot(self.inputs, np.transpose(self.weights)) + self.bias


if __name__ == '__main__':
    _input = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])
    w_at_h1 = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
    w_at_h2 = np.array([[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]])
    b1 = np.array([2.0, 3.0, 0.5])
    b2 = np.array([-1.0, 2.0, -0.5])

    mln = MultiLayerNeuralNetwork(inputs=_input, weights=w_at_h1, bias=b1)
    result_at_h1: np.array = mln.compute()
    print(result_at_h1)

    # pass the output of the layer1 as input for layer2
    mln.inputs = result_at_h1
    mln.weights = w_at_h2
    mln.bias = b2

    result_at_h2 = mln.compute()
    print(result_at_h2)
