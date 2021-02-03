import random
import numpy as np


class Neuron:
    def __init__(self, number_of_inputs):
        self.weights = []
        for _ in range(number_of_inputs + 1):
            self.weights.append(random.random())
        self.weights = np.array(self.weights)

    def forward(self, X):
        multi = np.multiply(self.weights, np.append(X, 1))
        suma = np.sum(multi)
        return relu(suma)


class Layer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = []
        for _ in range(number_of_neurons):
            self.neurons.append(Neuron(number_of_inputs))

    def forward(self, X):
        rsults = []
        for neuron in self.neurons:
            rsults.append(neuron.forward(X))
        return rsults


class Network:
    def __init__(self, layers_def, input_size):
        self.layers = []
        for i, layer in enumerate(layers_def):
            if not self.layers:
                self.layers.append(Layer(layer, input_size))
            else:
                self.layers.append(Layer(layer, layers_def[i-1]))

    def forward(self, X):
        result = 0
        for i, layer in enumerate(self.layers):
            if i == 0:
                result = layer.forward(X)
            else:
                result = layer.forward(result)
        return softmax(result)


def relu(X):
    return np.maximum(0, X)


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum