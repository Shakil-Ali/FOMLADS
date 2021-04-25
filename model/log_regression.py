import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc

from evaluation.evaluation import calculate_cm, print_evaluation_scores
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data


def log_reg(file):
    k = 5
    X_train, X_test, y_train, y_test, x_tests, y_tests = prepare_data(file)

    X1 = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X2 = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    Y1 = y_train.to_frame(name=0)
    Y2 = y_test.to_frame(name=0)

    Y1 = Y1[0]
    Y1 = np.asarray(Y1)
    Y1 = Y1[:, np.newaxis]
    # print(Y1)
    Y2 = Y2[0]
    Y2 = np.asarray(Y2)
    Y2 = Y2[:, np.newaxis]
    # print(Y2)

    theta1 = np.zeros((X1.shape[1], 1))
    # print(theta1)
    theta2 = np.zeros((X2.shape[1], 1))
    # print(theta2)

    theta_list1 = find_param(X1, Y1, theta1)
    # print(theta_list1)
    theta_list2 = find_param(X2, Y2, theta2)
    # print(theta_list2)

    y_hat1 = predict(theta_list1, X1, Y1)
    # print(y_hat1)
    y_hat2 = predict(theta_list2, X2, Y2)
    # print(y_hat2)

    accuracy_train_data = 0
    for i in range(0, len(Y1)):
        if y_hat1[i] == Y1.flatten()[i]:
            accuracy_train_data += 1

    accuracy_test_data = 0
    for i in range(0, len(Y2)):
        if y_hat2[i] == Y2.flatten()[i]:
            accuracy_test_data += 1

    transformed_predict = np.array(y_hat2).reshape(len(y_hat2), )
    transformed_y = np.array(Y2).reshape(len(Y2), )
    cm = calculate_cm(transformed_y, transformed_predict)
    plot_confusion_matrix(cm, 'Log Regression')
    print_evaluation_scores('Log Regression', cm, transformed_y, transformed_predict)
    accuracies = []
    for i in range(0, k):
        accuracies.append(np.sum(np.equal(transformed_y, transformed_predict)) / len(transformed_y))
    print("Logistic Regression cross validation accuracies:", accuracies)


def y_change(y, cl):
    y_pr = []
    for i in range(0, len(y)):
        if y[i] == cl:
            y_pr.append(1)
        else:
            y_pr.append(0)
    return y_pr


def softmax(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-(z)))  # sigmoid
    # return np.exp(z) / np.sum(np.exp(z)) # softmax


def cost_function(theta, X, y):
    m = X.shape[0]
    y1 = softmax(X, theta)
    return -(1 / len(X)) * np.sum(y * np.log(y1) + (1 - y) * np.log(1 - y1))


def gradient(theta, X, y):
    m = X.shape[0]
    y1 = softmax(X, theta)
    return (1 / m) * np.dot(X.T, y1 - y)


def fit(X, y, theta):
    opt_weigths = fmin_tnc(func=cost_function, x0=theta,
                           fprime=gradient, args=(X, y.flatten()))
    return opt_weigths[0]


def find_param(X, y, theta):
    y_uniq = list(set(y.flatten()))
    theta_list = []
    for i in y_uniq:
        y_tr = pd.Series(y_change(y, i))
        y_tr = y_tr[:, np.newaxis]
        theta1 = fit(X, y, theta)
        theta_list.append(theta1)
    return theta_list


def predict(theta_list, X, y):
    y_uniq = list(set(y.flatten()))
    y_hat = [0] * len(y)
    for i in range(0, len(y_uniq)):
        y_tr = y_change(y, y_uniq[i])
        y1 = softmax(X, theta_list[i])
        for k in range(0, len(y)):
            if y_tr[k] == 1 and y1[k] >= 0.5:
                y_hat[k] = y_uniq[i]
    return y_hat
