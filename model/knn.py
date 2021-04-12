import math

import pandas as pd
from evaluation.evaluation import print_evaluation_scores, calculate_cm
from plot.knn_visuals import k_value_graph
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data
from sklearn.neighbors import KNeighborsClassifier


def knn(file):
    X_train, X_test, y_train, y_test = prepare_data(file)
    # getting k
    k = math.ceil((math.sqrt(len(y_test))))
    # classify (professor confirmed we can use the sklearn classifier)
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = calculate_cm(y_test,y_pred)
    plot_confusion_matrix(cm, 'KNN')
    k_value_graph(X_train, y_train, X_test, y_test)
    print_evaluation_scores('KNN', cm, y_test, y_pred)
