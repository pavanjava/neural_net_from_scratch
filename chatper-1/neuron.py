import numpy as np
import math


def activate_single_neuron(inputs: [int], weights: [float], bias: int) -> [float]:
    return [inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias]


def activate_multiple_neuron(inputs, weights, bias) -> [float]:
    return [
        inputs[0] * weights[0][0] + inputs[1] * weights[0][1] + inputs[2] * weights[0][2] + bias[0],
        inputs[0] * weights[1][0] + inputs[1] * weights[1][1] + inputs[2] * weights[1][2] + bias[1],
        inputs[0] * weights[2][0] + inputs[1] * weights[2][1] + inputs[2] * weights[2][2] + bias[2],
    ]


# generic way of computing neuron outputs
def compute_neuron(inputs, weights, bias):
    layer_output = []
    for neuron_weights, neuron_bias in zip(weights, bias):
        neuron_output = 0
        for neuron_input, weights in zip(inputs, neuron_weights):
            neuron_output += neuron_input * weights
        neuron_output += neuron_bias
        layer_output.append(neuron_output)
    return layer_output


def activate(output):
    return [1 / (1 + math.pow(np.e, ele)) for ele in output]


# single neuron with 3 inputs, 3 weights and a bias
_inputs = [1, 2, 3]
_weights = [0.2, 0.8, -0.5]
_bias = 2
single_neuron_output = activate_single_neuron(inputs=_inputs, weights=_weights, bias=_bias)
print(single_neuron_output)
print(activate(single_neuron_output))

# a layer of 3 neurons with 3 inputs, 3 weights and a bias each

_inputs = [0.2, -0.8, -0.4]
_weights = [[0.2, 0.8, -0.5], [0.5, -0.9, -0.5], [0.2, 0.5, 0.3]]
_bias = [1, -0.5, 0.5]
multiple_neuron_output = activate_multiple_neuron(_inputs, _weights, _bias)
print(multiple_neuron_output)
print(activate(multiple_neuron_output))

output = compute_neuron(_inputs, _weights, _bias)
print(output)
print(activate(output))
