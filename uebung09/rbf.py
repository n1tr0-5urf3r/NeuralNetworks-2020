# Group: ---> Maximilian Bertsch, Fabian Ihle <---

import numpy as np
from numpy.linalg import pinv #pseudoinverse
import matplotlib.pyplot as plt


def output_func(x):
    """
    output function to predict using RBF
    :param x: a 1x1 or Nx1 input vector
    :return: cos(x)+0.5x
    """
    yx = lambda x: np.cos(x) + 0.5*x
    return yx(x)


def target_func(y, std):
    """
    Add random noise to the output function
    :param y: Nx1 vector
    :param std: standard dev for rand function
    :return: y_i + rand(-std,std)
    """
    for i in range(y.shape[0]):
        y[i] = y[i] + np.random.uniform(-std, std)
    return y

def L1_loss(y_true, y_pred):
    """

    :param y_true: N x 1 vector
    :param y_pred: N x 1 vector
    :return:
        sum of absolute difference between true and predicted values
    """
    n = y_pred.shape[0]
    sum = 0
    for i in range(n):
        sum += np.abs(y_true[i]- y_pred[i])
    return sum

class RBF(object):
    def __init__(self, w):
        """
        Initialize w & C
        :param w: Kx1 weight vector or list

        """
        self.w = np.array(w) if isinstance(w, list) else w
        self.c = np.zeros_like(self.w)

    def neurons_output(self, x, w):
        """
        compute the output of each RBF neuron, i.e exp(-(x_i-w_i)**2)/2) for all given x examples
        :param x: N x 1 -- N Points
        :param w: K x 1 -- K Neurons
        :return:
                 N x K  :
        """
        k = w.shape[0]
        n = x.shape[0]
        output = np.zeros((n, k))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = np.exp(-(x[i]-w[j])**2)/2
        return output

    def train(self, x, y):
        """
        given x, y and object weights  find the c using pseudo-inverse
        :param x: input x : (Nx1)
        :param y: output y to predict : (Nx1)
        :return:
        """
        output = self.neurons_output(x, self.w)
        pseudoinverse = np.linalg.pinv(output)
        self.c = np.dot(pseudoinverse, y)
        return self.c
    

    def predict(self, x):
        """
        given the trained RBF predict the function values
        for given inputs
        :param x: input x: (Nx1)
        :return: Kx1 predicted f(x) for each value...
        """
        sum = 0
        for i in range(self.c.shape[0]):
            sum += self.c[i]*np.exp(-((x-self.w[i])**2)/2)
        return sum


def evaluate_rbf(xtrain, ytrain, xtest, ytest, w):
    """
    train and test RBF for given w, training and test examples...
    :param xtrain: training points --> x
    :param ytrain: training labels --> f(x)
    :param xtest: testing --> x
    :param ytest: test --> f(x)
    :param w: weights to use
    :return:
    """
    rbf = RBF(w)
    rbf.train(xtrain, ytrain)
    ypred = rbf.predict(xtest)
    print(f"xtest:{xtest[:5]},ypred:{ypred[:5]}")
    print(f"Learned RBF coefficients = {rbf.c}")
    print(f"L1 Loss = {L1_loss(ypred, ytest)}")
    plt.plot(ypred)
    # plt.show()
    return ypred


if __name__ == "__main__":
    # training data and labels...
    xtrain = np.arange(-4, 6)  # sample training examples with xtrain

    # now  first generate f (x) and then add 
    # random noise sampled uniformly from the range [−0.2, 0.2] to each of the 10 values
    # using output_func and target_func
    # Generate f(x)

    # TODO: irgendwas stimmt mit xtrain, ytrain glaub noch nicht
    xtrain = output_func(xtrain)
    xtrain = target_func(xtrain, 0.2)

    #ytrain is the target lable, i.e. f(x)+random_noise
    ytrain = xtrain

    # test data and labels...
    xtest = np.linspace(-4, 6, num=100)
    ytest = output_func(xtest)    # Task c)

    # plot training and test curves...
    plt.figure()
    plt.plot(xtest, ytest)  # plot test data
    plt.plot(xtrain, ytrain, 'o')  # plot training data
    plt.xlabel('x')
    plt.ylabel('f(x)')
    #plt.show()

    # train RBF network by calling evaluate_rbf function
    # Train three diffferent networks using 3 different
    # set of weights and then plot them on a single plot 

    evaluate_rbf(xtrain, ytrain, xtest, ytest, [-4,-3,-2,-1])
    evaluate_rbf(xtrain, ytrain, xtest, ytest,[2,3,4,5])
    evaluate_rbf(xtrain, ytrain, xtest, ytest,[-4,-2,2,5])
    plt.show()


    # HINT: To have a single plot on the same figure
    # call plt.show only once at the end of plotting
    # all the plots.
    
