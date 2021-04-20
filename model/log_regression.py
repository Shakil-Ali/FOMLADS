import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize, fmin_tnc

from evaluation.evaluation import print_evaluation_scores, calculate_cm
from plot.knn_visuals import k_value_graph
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data

def log_reg(file):
    """
    ### FIRST METHOD ###

    #code goes here
    X_train, X_test, y_train, y_test = prepare_data(file)

    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)

    # print(len(X_train))
    # print(len(X_test))
    # print(len(y_train))
    # print(len(y_test))

    X1 = pd.concat([pd.Series(1, index=X_train.index, name='00'), X_train], axis=1)
    X2 = pd.concat([pd.Series(1, index=X_test.index, name='00'), X_test], axis=1)
    # print(X1)
    # print(X2)

    # print(y_test.unique())
    # print(y_train.unique())

    Y1 = np.zeros([X_train.shape[0], len(y_train.unique())])
    Y1 = pd.DataFrame(Y1)
    # print(Y1)
    
    Y2 = np.zeros([X_test.shape[0], len(y_test.unique())])
    Y2 = pd.DataFrame(Y2)
    # print(Y2)

    theta1 = np.zeros([X_train.shape[1]+1, Y1.shape[1]])
    theta1 = gradient_descent(X1, Y1, theta1, 0.02, 1500)
    print(theta1)

    # theta2 = np.zeros([X_test.shape[1]+1, Y2.shape[1]])
    # theta2 = gradient_descent(X2, Y2, theta2, 0.02, 1500)
    # print(theta2)

    # https://stackoverflow.com/questions/28735344/pythonvalueerror-shapes-3-and-118-1-not-aligned-3-dim-0-118-dim-0

    output = []
    print(output)
    for i in range(0, 3):
        print(f"i = {i}")
        # thetaVal1 = pd.DataFrame(theta1)
        thetaVal1 = np.array(theta1, dtype=object)
        print(thetaVal1)
        print("BEFORE H")
        h = hypothesis(thetaVal1[i], X1)
        print(h)
        print("AFTER H")
        output.append(h)
        print(output)
    output=pd.DataFrame(output)
    print(output)
    """

    X_train, X_test, y_train, y_test = prepare_data(file)

    # print(len(X_train))
    # print(len(X_test))
    # print(len(y_train))
    # print(len(y_test))

    X1 = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X2 = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    # print(X1)
    # print(X2)

    Y1 = y_train.to_frame(name=0)
    Y2 = y_test.to_frame(name=0)
    # print(Y1)
    # print(Y2)

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

    accuracy_train_data=0
    for i in range(0, len(Y1)):
        if y_hat1[i] == Y1.flatten()[i]:
            accuracy_train_data += 1
    print(f"Log Reg Training Accuracy = {accuracy_train_data/len(X1)*100}")

    accuracy_test_data=0
    for i in range(0, len(Y2)):
        if y_hat2[i] == Y2.flatten()[i]:
            accuracy_test_data += 1
    print(f"Log Reg Testing Accuracy = {accuracy_test_data/len(X2)*100}")



"""    
# FIRST METHOD FUNCTIONS 
def hypothesis(theta, X):
    return 1 / (1 + np.exp(-(np.dot(theta, X.T)))) - 0.0000001

def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))

def gradient_descent(X, y, theta, alpha, epochs):
    m = len(X)
    for i in range(0, epochs):
        for j in range(0, 3):
            theta = pd.DataFrame(theta)
            h = hypothesis(theta.iloc[:,j], X)
            for k in range(0, theta.shape[0]):
                theta.iloc[k, j] -= (alpha/m) * np.sum((h-y.iloc[:, j])*X.iloc[:, k])
            theta = pd.DataFrame(theta)
    return theta, cost

"""

def y_change(y, cl):
    y_pr=[]
    for i in range(0, len(y)):
        if y[i] == cl:
            y_pr.append(1)
        else:
            y_pr.append(0)
    return y_pr

def hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1/(1+np.exp(-(z))) # sigmoid 
    # return np.exp(z) / np.sum(np.exp(z)) # softmax
    

def cost_function(theta, X, y):
    m = X.shape[0]
    y1 = hypothesis(X, theta)
    return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))

def gradient(theta, X, y):
    m = X.shape[0]
    y1 = hypothesis(X, theta)
    return (1/m) * np.dot(X.T, y1 - y)

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
    y_hat = [0]*len(y)
    for i in range(0, len(y_uniq)):
        y_tr = y_change(y, y_uniq[i])
        y1 = hypothesis(X, theta_list[i])
        for k in range(0, len(y)):
            if y_tr[k] == 1 and y1[k] >= 0.5:
                y_hat[k] = y_uniq[i]
    return y_hat


