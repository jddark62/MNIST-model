import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv')


data = np.array(data) # convert to numpy array

m, n = data.shape # m = number of examples, n = number of features

np.random.shuffle(data) # shuffle the data
#why shuffle the data? because we want to split the data into training and testing data
#and we want to make sure that the training data and testing data are not biased

data_dev = data[0:1000].T # dev set
Y_dev = data_dev[0] # dev set labels
X_dev = data_dev[1:n] # dev set features

# normalize the data
X_dev = X_dev/255

data_train = data[1000:m].T # training set
Y_train = data_train[0] # training set labels
X_train = data_train[1:n] # training set features

# normalize the data
X_train = X_train/255

_, m_train = X_train.shape # number of training examples

# see how Y_train looks like
print(Y_train)

# neural network has two-layer architecture
# layer 1: input layer (784 neurons)
# layer 2: hidden layer (10 units) (ReLU activation)
# layer 3: output layer (10 units) (softmax activation)

# initialize the parameters
def initialize_parameters():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

# ReLU activation function
def relu(Z):
    A = np.maximum(0, Z)
    return A

# softmax activation function
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# forward propagation
def forward_propagation(X, parameters):
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = softmax(Z2)
    cache = {"Z1": Z1, 
            "A1": A1,
            "Z2": Z2,
            "A2": A2}
    return A2, cache

# ReLU activation function derivative
def relu_derivative(Z):
    dZ = np.array(Z, copy=True)
    dZ[Z <= 0] = 0
    dZ[Z > 0] = 1
    return dZ

# one-hot encoding
# for example, if Y = 3, then Y_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(Y):
    Y_one_hot = np.zeros((10, Y.shape[0]))
    for i in range(Y.shape[0]):
        Y_one_hot[Y[i], i] = 1
    return Y_one_hot

#backward propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    Y_one_hot = one_hot(Y)
    dZ2 = cache['A2'] - Y_one_hot
    dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(parameters['W2'].T, dZ2) * relu_derivative (cache['Z1'])
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    



