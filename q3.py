import numpy as np
from sklearn.metrics import mean_squared_error
from q1 import NeuralNetwork


def run(input, hid, outp, eta, it):
    a = NeuralNetwork(input, hid, outp, eta)
    a.fit(X, y, it)
    print('-' * 50)
    print("Input: " + str(X))
    print("Actual Output: " + str(y))
    print("\n")
    print("Predicted Output: " + str(a.predict(X)))
    a.plot()


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

it = 10
for i in range(5):  # diff iter
    run(2, 2, 1, 0.5, it)
    it = it * 10

"""it = 0.1
for i in range(10): # diff learning
    run(2, 2, 1, it, 1000)
    it += it"""

"""a = NeuralNetwork(2, 2, 1, 0.9) # b 
a.fit(X, y, 2000)
a.plot()"""

"""for i in range(2,6):
    run(2,i,1,0.9,1000)"""
