"""
Group: ---> Maximilian Bertsch, Fabian Ihle <---


Your tasks:
- Fill in your names.
- Complete the methods at the marked code blocks.
- Please comment on the most important places.

Test your implementation at least with:
    perceptron.py aImpliesbinput.txt oroutput.txt 1000 0.001
    perceptron.py xnorinput.txt xoroutput.txt 1000 0.001
    perceptron.py test1input.txt test1output.txt 1000 0.001

Have fun!
"""

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import random


class Perceptron:
    """
    The class Perceptron implements a single-layer perceptron
    with an arbitrary input dimension and binary step function
    as activation. The output is either 1 or 0
    """

    def __init__(self, dimension):
        """
        initializes a new Perceptron object
        :param dimension (int): dimension of the input data
        """
        self._dimension = dimension

        # random initialization of weights between -0.1 and 0.1:
        self._weights = np.zeros([self._dimension])

        # Initialize with random values between -0.1 and 0.1
        for i, w in enumerate(self._weights):
            self._weights[i] = random.uniform(-0.1, 0.1)

        # Bias that has to be added to the activation.
        self._bias_weight = -1.0

    def predict(self, input_: np.matrix) -> int:
        """
        Calculates the output of the perceptron for an input vector
        :param input_: input vector
        :return: network output (1 or 0)
        """
        inputAll = np.dot(input_, self._weights)
        # Sum of all input vectors > threshold? 1 else 0
        if inputAll + self._bias_weight >= 0:
            return 1
        return 0

    def _train_pattern(self, input_: np.matrix, target: int, lr: float):
        """
        If the output does not correspond to the desired value,
        the weights and bias are changed according to the delta rule.

        :param input: input vector
        :param target: desired output value (1 or 0)
        :param lr: the learning rate of the perceptron learning rule
        """

        result = self.predict(input_)
        if result != target:
            # Calculate delta w_ji
            # We used the delta learning rule from NN 07 Backpropagation, Slide 9 "online version of the Delta rule" for single layer networks
            delta_wji = lr * input_ * (target - result)
            # Convert ndarray input to matrix
            self._weights_matrix = np.asmatrix(self._weights)
            #delta_wji = np.array(delta_wji.T)[0]

            # Use delta bias weight as an on neuron and alter it
            delta_bias_wji = lr * 1 * (target - result)
            self._bias_weight += delta_bias_wji
            self._weights_matrix += delta_wji
        pass

    def train(self, input_vectors: np.matrix, targets: np.array, epochs: int, lr: float) -> list:
        """
        Trains the perceptron for a set of input vector over a specified number of epochs.
        Returns the number of incorrectly classified training vectors.
        :param input_vectors: two dimensional matrix with training vectors as rows.
                input_vectors[0,:] is the first training vector, input_vectors[1,:] the second...
        :param targets: target outputs. targets[0] is the  desired output
                for the first training vector.
        :param epochs: Number of epochs.
        :param lr: the learning rate of the perceptron learning rule
        :return: average error rate for every epoch
        """

        error_rates = []
        for i in range(0, epochs):
            iterations = 0
            errorcount = 0
            for j in range(0, input_vectors.shape[0]):
                iterations += 1
                self._train_pattern(input_vectors[j, :], targets[j], lr)
                # split training and predition in two loops
                # first train all input then predict all
                result = self.predict(input_vectors[j, :])
                if result != targets[j]:
                    errorcount += 1
            error_rates.append(errorcount/iterations)

        # return [0]*iterations
        return error_rates


def read_double_array(filename) -> np.array:
    """
    reads data from the provided files.
    :param filename: path to a file
    :return: np.array of the matrix given in the file
    """
    with open(filename) as file:
        content = file.readlines()
        # remove first line
        content = content[1:]
        return np.loadtxt(content)


def read_double_matrix(filename):
    """
    reads data from the provided files.
    :param filename: path to a file
    :return: np.matrix of the matrix given in the file
    """
    return np.matrix(read_double_array(filename))


def main():
    parser = argparse.ArgumentParser("perceptron")
    parser.add_argument('inputs', type=str)
    parser.add_argument('outputs', type=str)
    parser.add_argument('epochs', type=int)
    parser.add_argument('lr', type=float)
    args = parser.parse_args()

    print('arguments:', args)
    input_vectors = read_double_matrix(args.inputs)
    targets = read_double_array(args.outputs)
    perceptron = Perceptron(np.size(input_vectors[0]))

    errors = perceptron.train(input_vectors, targets, args.epochs, args.lr)


if __name__ == "__main__":
    main()
