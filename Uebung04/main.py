"""
Group: ---> (FILL IN YOUR NAMES) <---


Your tasks:
Fill in your name.
Please complete the methods at the marked code blocks.
Please comment on the most important places.
Test your implementation at least with:

perceptron.py aImpliesbinput.txt oroutput.txt 1000 0.001
perceptron.py xnorinput.txt xoroutput.txt 1000 0.001
and
perceptron.py test1input.txt test1output.txt 1000 0.001

Have fun!

"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from perceptron import *


def calculate_relationship_table(argv, lr_list , epochs_list):
    input_vectors = read_double_matrix(argv[1])
    targets = read_double_array(argv[2])

    relationship = np.zeros((len(lr_list), len(epochs_list)))
    # lr_list = [1e-1, 1e-2, 1e-3, 1e-4]
    # epochs_list = [10, 25, 50, 100]
    for i, learningrate in enumerate(lr_list):
        for j, iterations in enumerate(epochs_list):
            perceptron = Perceptron(np.size(input_vectors[0]))

            errors = perceptron.train(input_vectors, targets, iterations, learningrate)
            relationship[i, j] = errors[-1]
    print(f"This table shows error values for learning rates (rows) vs number of epochs (columns)")
    print(relationship)

    print(f"Please press enter to continue.....")
    input()

def main(argv,threshold):

    input_vectors = read_double_matrix(argv[1])
    targets = read_double_array(argv[2])
    iterations = int(argv[3])
    learningrate =  float(argv[4])

    perceptron = Perceptron(np.size(input_vectors[0]))

    errors= perceptron.train(input_vectors,targets,iterations,learningrate)

    plt.plot(errors, 'r')
    plt.ylabel('E(t)')
    plt.xlabel('epoch')
    plt.show()

    print("the final error is: " + str(errors[-1]))
    if (errors[-1] < threshold) :
        print("Yeah, correctly learned the dataset ",argv[1])
    else:
        print("Oh no, not correctly learned the dataset ",argv[1])        

if __name__ == "__main__":
    main(["","aImpliesbinput.txt","aImpliesboutput.txt","1000", "0.001"],0.1);
    main(["","xnorinput.txt","xnoroutput.txt","1000", "0.001"],0.3);
    main(["","test1input.txt","test1output.txt","100", "0.001"],0.1);

    # Please uncomment the following lines to check learning and epoch relationship for different combinations
    calculate_relationship_table(["","aImpliesbinput.txt","aImpliesboutput.txt"],[1e-1, 1e-2, 1e-3, 1e-4],[100, 250, 500, 1000])
    # calculate_relationship_table(["","xnorinput.txt","xnoroutput.txt"],[1e-1, 1e-2, 1e-3, 1e-4],[100, 250, 500, 1000])
    # calculate_relationship_table(["","test1input.txt","test1output.txt"],[1e-1, 1e-2, 1e-3, 1e-4],[100, 250, 500, 1000])

