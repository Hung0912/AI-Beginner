'''SVM:
 hyperplane that separates +ve and -ve examples
 with the largest margin
 while keeping the misclassification as low as possible
'''

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

from read_data import *


def cost(W, X, Y):
    N = X.shape[0]

    distances = 1 - Y*(np.dot(X, W))
    # print(distances)
    distances[distances < 0] = 0
    h_loss = C*(np.sum(distances) / N)

    cost = 1/2 * np.dot(W, W) + h_loss
    return cost


def gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (C * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


# def gradient_descent(X, Y, weights_init, learning_rate, tolerate):
#     '''
#     vanila gradient descent
#     '''
#     interation = 0
#     weights = [weights_init]
#     N = X.shape[0]
#     while True:
#         interation += 1
#         n_weight = weights[-1] - learning_rate*gradient(weights[-1], X, Y)
#         weights.append(n_weight)
#         # When to stop
#         if np.linalg.norm(gradient(weights[-1], X, Y))/N <= tolerate:
#             print(np.linalg.norm(gradient(weights[-1], X, Y))/N)
#             break

#     return weights, interation


def gradient_descent_SGD(features, outputs, learning_rate, max_epochs, cost_threshold=0.01):
    '''
    with SGD caculate gradient value by only one examples at a time
    '''
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            grad = gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * grad)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            _cost = cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, _cost))
            # stoppage criterion
            if abs(prev_cost - _cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = _cost
            nth += 1
    return weights


def train(X_train, Y_train):
    print('Start training model...')
    W = gradient_descent_SGD(
        X_train, Y_train, learning_rate, max_epochs, cost_threshold)
    print('Train finished!')
    return W


if __name__ == "__main__":
    # read data
    X, Y = read_data('data.csv')

    # split data for training and testing 80:20
    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)
    print("splitting dataset into train and test sets...")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # regulaziration parameter
    C = 10000
    learning_rate = 0.000001
    max_epochs = 5000
    cost_threshold = 0.01

    W = train(X_train.to_numpy(), Y_train.to_numpy())

    # testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)
    print("accuracy on test dataset: {}".format(
        accuracy_score(Y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(
        recall_score(Y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(
        recall_score(Y_test.to_numpy(), y_test_predicted)))
