import numpy as np
from data_generator import generate_data


class DenseLayer:
    # initialize the weights and biases randomly
    def __init__(self, no_of_inputs: int, no_of_neurons: int):
        self.weights = 0.001 * np.random.randn(no_of_inputs, no_of_neurons)
        self.bias = np.zeros((1, no_of_neurons))

    # forward pass to compute the Wx + b, eventually this is fead to activation functions in later chapters
    def forward(self, inputs: np.array) -> np.array:
        return np.dot(inputs, self.weights) + self.bias


# driver code which generate the output
X, y = generate_data()
dense1 = DenseLayer(no_of_inputs=2, no_of_neurons=50)
response = dense1.forward(inputs=X)
print(response[:5])
