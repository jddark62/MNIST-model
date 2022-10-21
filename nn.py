import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

# ReLU activation function and its derivative
def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

# softmax activation function
def softmax(Z):
    # Compute softmax values for each sets of scores in x.
    # Softmax = exp(Z - max(Z)) / sum(exp(Z - max(Z))
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

# initialization of parameters W,b with random numbers
def init_params(size):
    W1 = np.random.rand(10,size) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2

# forward propagation
def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

# one hot vector format
def one_hot(Y):
    # one hot encoding = 0 vector with 1 only in the position corresponding to the value in Y
    one_hot_Y = np.zeros((Y.max()+1, Y.size)) # change order but not nbr
    one_hot_Y[Y, np.arange(Y.size)] = 1 # Y.max()+1 is nb of rows
    return one_hot_Y

# backward propagation
def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

# update weights and biases
def update_weights(W1,W2,b1,b2,dW1,db1,dW2,db2,alpha):# alpha < lr
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))
    
    return  W1, W2, b1, b2

def predictions(A2): # A2: output matrix of the network on test set
    return np.argmax(A2,0)

# get accuracy
def accuracy(predictions,Y): # predictions: vector of predictions, Y: vector of labels
    return np.mean(predictions == Y)

# gradient descent
def model(X, Y, alpha, iterations):
    size, m = X.shape #size = (784, 60000)

    (W1, b1, W2, b2) = init_params(size)
    for i in range(iterations):
        (Z1, A1, Z2, A2) = forward_propagation(X,W1,b1,W2,b2)
        (dW1, db1, dW2, db2) = backward_propagation(X, Y, A1, A2, W2, Z1, m)
        (W1, W2, b1, b2) = update_weights(W1,W2,b1,b2,dW1,db1,dW2,db2,alpha)
        
        if (i+1) % int(iterations/10) == 0:
            print("Iteration: ", i+1, " Cost: ", np.mean(np.square(A2 - one_hot(Y))))
            prediction = predictions(A2)
            print("Accuracy: ", accuracy(prediction,Y))
    return W1, W2, b1, b2

# make predictions
def predict(X, W1, W2, b1, b2):
    _, _, _, A2 = forward_propagation(X,W1,b1,W2,b2)
    return predictions(A2)

# show predictions
def show_predictions(X, Y, W1, W2, b1, b2):
    prediction = predict(X, W1, W2, b1, b2)
    print("Accuracy: ", accuracy(prediction,Y))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(X[:,i].reshape(28,28), cmap='gray')
        plt.title("Prediction: " + str(prediction[i]))
        plt.axis('off')
    plt.show()

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# scale factor to normalize data
scale_factor = 255

# reshape data
width = X_train.shape[1]
height = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], width * height) / scale_factor
X_test = X_test.reshape(X_test.shape[0], width * height) / scale_factor

# model parameters
W1, W2, b1, b2 = model(X_train.T, Y_train, 0.15, 200)
#dump file for trained model
pickle.dump((W1, W2, b1, b2), open("model.p", "wb"))

#load dump file
(W1, W2, b1, b2) = pickle.load(open("model.p", "rb"))

# show predictions
show_predictions(X_test.T, Y_test, W1, W2, b1, b2)
show_predictions(X_train.T, Y_train, W1, W2, b1, b2)
show_predictions(X_train.T[:,0:10], Y_train[0:10], W1, W2, b1, b2)
show_predictions(X_train.T[:,10:20], Y_train[10:20], W1, W2, b1, b2)

""" # increasing the number of iterations
W1, W2, b1, b2 = model(X_train.T, Y_train, 0.15, 2000)
show_predictions(X_test.T, Y_test, W1, W2, b1, b2) """

# increasing accuracy by decreasing the learning rate
W1, W2, b1, b2 = model(X_train.T, Y_train, 0.05, 2000)
show_predictions(X_test.T, Y_test, W1, W2, b1, b2)

"""
how does learning rate affect the accuracy of the model?
answer : the accuracy of the model is inversely proportional to the learning rate
how does the number of iterations affect the accuracy of the model?
answer : the accuracy of the model is directly proportional to the number of iterations
"""