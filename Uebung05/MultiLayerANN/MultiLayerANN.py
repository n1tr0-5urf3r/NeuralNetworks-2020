"""
Group: ---> Maximilian Bertsch, Fabian Ihle <---

Your tasks:
    Fill in your names.
    Complete the methods at the marked code blocks.
    Please comment on the most important places.

run your program with
    python3 MultiLayerANN.py --key1=value1 --key2=value2 ...
e.g.
    python3 MultiLayerANN.py --epochs=1000 --inputs='digits-input.txt' --outputs='digits-target.txt' --activation='tanh'
W
Key/value pairs that are unchanged use their respective defaults.
"""
import matplotlib

matplotlib.use('TkAgg')

import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


class Sigmoid:
    @staticmethod
    def f(x: np.array) -> np.array:
        """ the sigmoid function """
        sigmoid = lambda y: 1 / (1 + np.exp(-y))
        return sigmoid(x)

    @staticmethod
    def d(x: np.array) -> np.array:
        """ the first derivative """
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))


class TanH:
    @staticmethod
    def f(x: np.array) -> np.array:
        """ the tanh function """
        tanh = lambda y: (np.exp(y) - np.exp(-y)) / (np.exp(y) + np.exp(-y))
        return tanh(x)

    @staticmethod
    def d(x: np.array) -> np.array:
        """ the first derivative """
        return 1 - (TanH.f(x) ** 2)


class MultiLayerANN:
    """
    The class MultiLayer implements a multi layer ANN with a flexible number of hidden layers.
    Backpropagation is used to calculate the gradients.
    """

    # the activation of the Bias neuron
    BIAS_ACTIVATION = -1

    def __init__(self, act_fun, *layer_dimensions):
        """
        initializes a new MultiLayerANN object
        :param act_fun: Which activation function to use.
        :param layer_dimensions: each parameter describes the amount of neurons in the corresponding layer.
        E.g. MultiLayerANN(TanH, 3, 10, 4) creates a network with 3 layers, 3 input_ neurons, 10 hidden neurons,
            4 output neurons, and uses the tanh activation function.
        """
        if len(layer_dimensions) < 2:
            raise Exception("At least an input_ and output layer must be given")
        self._act_fun = act_fun
        self._layer_dimensions = layer_dimensions

        # the net_input value for each non input_ neuron,
        # each list element represents one layer
        self._net_inputs = []

        # Type: list of np.arrays. The activation value for each non input_ neuron,
        # each list element represents one layer
        self._activations = []

        # Type: list of np.arrays. The back propagation delta value for each non input_ neuron,
        # each list element represents one layer
        self._deltas = []

        # Type: list of np.arrays. List of all weight matrices. 
        # Weight matrices are randomly initialized between -1 and 1
        self._weights = []

        # Type: list of np.arrays. List of all delta weight matrices. 
        # They are added to the corresponding weight matrices after each training step
        self._weights_deltas = []

        for i in range(len(self._layer_dimensions[1:])):
            layer_size, prev_layer_size = self._layer_dimensions[i + 1], self._layer_dimensions[i]
            self._net_inputs.append(np.zeros(layer_size))
            self._activations.append(np.zeros(layer_size))
            self._deltas.append(np.zeros(layer_size))

            # we use +1 to consider the bias-neurons
            # weights are chosen randomly (uniform distribution) between -1 and 1
            self._weights.append(np.random.rand(prev_layer_size + 1, layer_size) * 2 - 1)
            self._weights_deltas.append(np.zeros([prev_layer_size + 1, layer_size]))

    def _predict(self, input_: np.array) -> np.array:
        """
        calculates the output of the network for one input vector
        :param input_: input vector
        :return: activation of the output layer
        """

        # Calculate the first (input) layer with identity function, bias and its weights
        input_ = np.append(input_, self.BIAS_ACTIVATION)
        output = np.dot(input_, self._weights[0])
        self._net_inputs[0] = output

        # Calculate outputs for each hidden layer
        # Go 'till dimensions - 1 as the output layer will be calculated outside separatly
        for i in range(0, len(self._activations) - 1):
            # Add bias neuron to previous output vector
            output = self._act_fun.f(output)
            self._activations[i] = output
            newNetInput = np.append(output, self.BIAS_ACTIVATION)
            self._net_inputs[i+1] = np.dot(newNetInput, self._weights[i+1])

        # Calculate output layer
        self._activations[-1] = self._act_fun.f(self._net_inputs[-1])
        return self._activations[-1]

    def _train_pattern(self, input_: np.array, target: np.array, lr: float, momentum: float, decay: float):
        """
        trains one input vector
        :param input_: one input vector
        :param target: one target vector
        :param lr: learning rate
        :param momentum:
        :param decay: weight decay:
        """
        self._predict(input_)

        # - first compute output layer deltas
        self._deltas[-1] = self._act_fun.d(self._net_inputs[-1]) * (target - self._activations[-1])
        # - then compute hidden layer deltas, consider that no delta is needed for the bias neuron
        #   Note: self._deltas[0:-1] = ignore last delta
        for delta, last_delta, weights, net_inputs in zip(reversed(self._deltas[0:-1]), reversed(self._deltas[1:]),
                                                          reversed(self._weights[1:]),
                                                          reversed(self._net_inputs[0:-1])):
            newDelta = self._act_fun.d(net_inputs) * np.dot(weights[0:-1], last_delta)
            self._deltas[self._deltas.index(delta)] = newDelta

        # add input layer activations to activations and ignore output layer activations
        act_with_input = [input_] + self._activations[0:-1]
        # apply the weight matrice to our input vector, ignore bias neuron
        for weights_layer, weight_deltas_layer, activation_layer, delta_layer in zip(self._weights,
                                                                                     self._weights_deltas,
                                                                                     act_with_input, self._deltas):
            newDeltaMatrix = lr * (np.outer(activation_layer, delta_layer)) + momentum * weight_deltas_layer[0:-1]
            newDeltaMatrix = np.vstack([newDeltaMatrix, np.zeros(newDeltaMatrix.shape[1])])
            # Apply row of zeros for no weight change with bias neuron and to match dimensions
            self._weights_deltas[self._weights_deltas.index(weight_deltas_layer)] = newDeltaMatrix
            # Update weight matrix
            self._weights[self._weights.index(weights_layer)] = weights_layer + newDeltaMatrix


    def train(self, inputs: [np.array], targets: [np.array], epochs: int, lr: float, momentum: float,
              decay: float) -> list:
        """
        trains a set of input_ vectors. The error for each epoch gets printed out.
        In addition, the amount of correctly classiefied input_ vectors in printed
        :param inputs: list of input_ vectors
        :param targets: list of target vectors
        :param epochs: number of training iterations
        :param lr: learning rate for the weight update
        :param momentum: momentum for the weight update
        :param decay: weight decay for the weight update
        :return: list of errors. One error value for each epoch. One error is the mean error over all input_vectors
        """
        errors = []
        for epoch in range(epochs):
            error = 0
            for input_, target in zip(inputs, targets):
                self._train_pattern(input_, target, lr, momentum, decay)
            for input_, target in zip(inputs, targets):
                output = self._predict(input_)
                d = (target - output)
                error += np.dot(d.T, d) / len(d)
            error /= len(inputs)
            errors.append(error)
            print("epoch: {0:d}   error: {1:f}".format(epoch, float(error)))

        print("final error: {0:f}".format(float(errors[-1])))
        # evaluate the prediction
        correct_predictions = 0
        for input_, target in zip(inputs, targets):
            # for one output use a threshold with 0.5
            if isinstance(target, float):
                correct_predictions += 1 if np.abs(self._predict(input_) - target) < 0.5 else 0

            # for multiple outputs choose the outputs with the highest value as predicted class
            else:
                prediction = self._predict(input_)
                predicted_class = np.argmax(prediction)
                correct_predictions += 1 if target[predicted_class] == 1 else 0
        print("correctly classified: {0:d} / {1:d}".format(correct_predictions, len(inputs)))
        return errors


def read_double_array(filename):
    """
    reads an np.array from the provided file.
    :param filename: path to a file
    :return: np.array of the matrix given in the file
    """
    with open(filename) as file:
        content = file.readlines()
        return np.loadtxt(content)


def main():
    parser = argparse.ArgumentParser("MultiLayerANN")
    parser.add_argument('--inputs', type=str, default='digits-input.txt')
    parser.add_argument('--outputs', type=str, default='digits-target.txt')
    #parser.add_argument('--inputs', type=str, default='evenParity_input.txt')
    #parser.add_argument('--outputs', type=str, default='evenParity_target.txt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--activation', type=str, default='sigmoid', help='sigmoid or tanh')
    #parser.add_argument('--activation', type=str, default='tanh', help='sigmoid or tanh')
    args = parser.parse_args()

    # or be lazy, don't pass runtime args and change them here, e.g.
    # v = 'digits-input.txt', 'digits-target.txt', 100, 0.1, 0.0, 0.0, 'tanh'
    # args.inputs, args.outputs, args.epochs, args.lr, args.momentum, args.decay, args.activation = v

    print(args)

    input_vectors = read_double_array(args.inputs)
    targets = read_double_array(args.outputs)
    fun = Sigmoid if args.activation.lower() == 'sigmoid' else TanH
    num_outputs = 1 if isinstance(targets[0], float) else targets.shape[1]

    net = MultiLayerANN(fun, input_vectors.shape[1], 20, num_outputs)
    errors = net.train(input_vectors, targets, args.epochs, args.lr, args.momentum, args.decay)

    plt.plot(errors, 'r')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.grid()
    plt.show()
    plt.savefig('train.png')


if __name__ == "__main__":
    main()
