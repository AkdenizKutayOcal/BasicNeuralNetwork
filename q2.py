"""
@author: Akdeniz Kutay Ocal
Title: CMPE442_Assignment3_Q2
Description: Question 2 - AND function
"""

import numpy as np
from q1 import NeuralNetwork

# AND function instances input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [0], [0], [1]])

inSize = 2
clsSize = 1


def tryEtas(startEta, endEta, iterNo):
    """
    Trains the model for different learning rates and prints their
    errors after given number of iterations.
    @param startEta: float
    @param endEta: float
    @param iterNo: int
    @return: None
    """
    print("Trying learning rates starting from " + str(startEta) + " to " + str(endEta))
    eta = startEta
    while eta <= endEta:
        print('-' * 50)
        print("Learning for eta = " + str(eta))

        nn = NeuralNetwork(inSize, 2, clsSize, eta)
        nn.fit(X, y, iterNo)

        print("Input: " + str(X))
        print("Actual Output: " + str(y))
        print("Predicted Output: " + str(nn.predict(X)))
        eta += 0.1


def tryIterNo(startIter, endIter, eta):
    """
    Trains the model for different iteration numbers with given eta
    and prints their errors.
    @param startIter: int
    @param endIter: int
    @param eta: float
    @return: None
    """
    print("Trying number of iterations starting from " + str(startIter) + " to " +
          str(endIter) + " with eta= " + str(eta))

    iterNo = startIter

    while iterNo <= endIter:
        print('-' * 50)
        print("Learning for iterNo = " + str(iterNo))

        nn = NeuralNetwork(inSize, 2, clsSize, eta)
        nn.fit(X, y, iterNo)

        print("Input: " + str(X))
        print("Actual Output: " + str(y))
        print("Predicted Output: " + str(nn.predict(X)))
        iterNo += 1000


def plotWith(iterNo, eta, sl):
    """
    Plot error - # of iterations function with given parameters
    @param iterNo: int
    @param eta: float
    @param sl: int
    @return: None
    """
    nn = NeuralNetwork(inSize, sl, clsSize, eta)
    nn.fit(X, y, iterNo)
    nn.plot()


# tryEtas(0.1, 1, 1000)
# tryIterNo(1000,10000,0.1)
plotWith(1000, 0.4, 2)

# TODO
# try different hidden layer unit sizes
