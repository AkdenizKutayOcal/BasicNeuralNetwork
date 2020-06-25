"""
@author: Akdeniz Kutay Ocal
Title: CMPE442_Assignment3_Q1
Description: Question 1 - Neural Network Class
"""

# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, inSize, sl2, clsSize, lrt):
        """
        Initialize the neural network model.
        @param inSize: int
        @param sl2: int
        @param clsSize: int
        @param lrt: float
        """

        self.iSz = inSize  # number of input units
        self.oSz = clsSize  # number of output units
        self.hSz = sl2  # number of hidden units

        # initialize weights while last column is biases
        self.weights1 = (np.random.rand(self.hSz, self.iSz + 1) - 0.5) / np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz, self.hSz + 1) - 0.5) / np.sqrt(self.hSz)

        self.eta = lrt  # learning rate

    def sigmoid(self, x, deriv=False):
        """
        Apply sigmoid function by default, take derivative if deriv is True.
        @param x: np.array
        @param deriv: boolean
        @return: np.array
        """
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def sum(self, z1, b):
        """
        Sum z1 elements with a bias.
        @param z1: np.array
        @param b: np.array
        @return: np.array
        """
        for i in range(len(z1)):
            # print(z1[0][i], float(b[i]))
            z1[i] += float(b[i])
            # print(z1[0][i], float(b[i]))
        return z1

    def feedforward(self, x):
        """
        Activation's of each neuron are computed until output has reached.
        Non generic! Just works for one hidden layer. For single input example.
        @param x: np.array
        @return: None
        """
        # Computing activations of units of hidden layer
        self.z1 = self.sum(np.dot(self.weights1[:, :-1], x), self.weights1[:, [-1]])
        self.a1 = self.sigmoid(self.z1)

        # Computing activations of output units
        z2 = np.dot(self.weights2[:, :-1], self.a1) + self.weights2[:, [-1]]
        self.output = self.sigmoid(z2)

    def backprop(self, x, trg):
        """
        Computes the gradient of the loss function with respect to the weights
        and biases of the network for a single input–output example.
        @param x: np.array
        @param trg: int/float
        @return: np.array, np.array
        """

        # Calculating delta for weights and biases separately then combining them together

        # Delta2 calculations
        delta2_w = np.dot(np.reshape(self.a1, (self.hSz, 1)),
                          (2 * (trg - self.output) * self.sigmoid(self.output, True)))

        delta2_b = np.dot(1, (2 * (trg - self.output) * self.sigmoid(self.output, True)))

        # Delta 1 calculations
        delta1_w = np.dot(np.reshape(x.T, (self.iSz, 1)),
                          (np.dot(2 * (trg - self.output) * self.sigmoid(self.output, True),
                                  self.weights2[:, :-1]) * self.sigmoid(self.a1, True))).T

        delta1_b = np.dot(1, (np.dot(2 * (trg - self.output) * self.sigmoid(self.output, True),
                                     self.weights2[:, :-1]) * self.sigmoid(self.a1, True))).T

        # Concatenating biases to the ending columns
        delta1 = np.c_[delta1_w, delta1_b]
        delta2 = np.c_[delta2_w.T, delta2_b]

        return delta1, delta2

    def fit(self, X, y, iterNo):
        """
        Used for training the model with iterNo epochs. Batch approach is used.
        Calculates the error for every 100 iterations.
        @param X: np.array
        @param y: np.array
        @param iterNo: int
        @return: None
        """

        m = np.shape(X)[0]
        self.error_arr = []  # errors of every 100 iteration kept to plot
        self.it_arr = []    # number of iterations kept to plot
        error_total = 0.0

        for i in range(iterNo+1):

            D1 = np.zeros(np.shape(self.weights1))
            D2 = np.zeros(np.shape(self.weights2))
            for j in range(m):

                self.feedforward(X[j])
                [delta1, delta2] = self.backprop(X[j], y[j])
                D1 = D1 + delta1
                D2 = D2 + delta2

                # total error of all x instances for every 100 iterations
                if i % 100 == 0:
                    error_total += mean_squared_error(y[j], self.output)

            if i % 100 == 0:
                avg = error_total / m
                self.error_arr.append(avg)
                self.it_arr.append(i)
                print("Error of Epoch " + str(i) + " is: " + str(avg))
                error_total = 0.0

            # update weight and bias values
            self.weights1 = self.eta * (D1 / m) + self.weights1
            self.weights2 = self.eta * (D2 / m) + self.weights2


    def predict(self, X):
        """
        Predicts output of given parameters using learnt weights
        @param X: np.array
        @return: np.array
        """
        m = np.shape(X)[0]
        y = np.zeros(m)
        for i in range(m):
            self.feedforward(X[i])
            y[i] = self.output
        return y

    def plot(self):
        """
        Plots error - #of iterations graph
        @return: None
        """
        plt.title("Title", size=15)  # TODO
        plt.xlabel("Number of Iterations", color="blue", size=12)
        plt.ylabel("Error", color="blue", size=12)
        plt.plot(self.it_arr, self.error_arr, ".b")

        plt.show()