import math

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from evaluation.evaluation import print_evaluation_scores, calculate_cm
from plot.knn_visuals import k_value_graph
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data


def knn(file):
    X_train, X_test, y_train, y_test, x_tests, y_tests = prepare_data(file)
    # getting k
    k = math.ceil((math.sqrt(len(y_test))))
    # classify (professor confirmed we can use the sklearn classifier)
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    accuracies = []
    for i in range(0, len(x_tests)):
        y_pred = classifier.predict((x_tests[i]))
        acc = np.sum(np.equal(y_tests[i], y_pred)) / len(y_tests[i])
        accuracies.append(acc)
    y_pred = classifier.predict(X_test)
    cm = calculate_cm(y_test, y_pred)
    plot_confusion_matrix(cm, 'KNN')
    k_value_graph(X_train, y_train, X_test, y_test)
    print_evaluation_scores('KNN', cm, y_test, y_pred)
    print("KNN cross validation accuracies", accuracies)
