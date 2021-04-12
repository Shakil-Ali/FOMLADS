import math
import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix


from matplotlib import pyplot as plt
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def knn(file):
    X_train, X_test, y_train, y_test = prepare_data(file)
    # getting k
    k = math.ceil((math.sqrt(len(y_test))))
    # classify (professor confirmed we can use the sklearn classifier)
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'KNN')

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_NoSkLearn = np.sum(np.equal(y_test,y_pred))/len(y_test)

    # plotting confusion again to get precision and recall
    cm = confusion_matrix(y_test, y_pred)

    # Without Sklearn - getting recall and precision from confusion matrix 
    recall_NoSum = np.diag(cm) / np.sum(cm, axis = 1)
    precision_NoSum = np.diag(cm) / np.sum(cm, axis = 0)
    precision = np.mean(precision_NoSum)
    recall = np.mean(recall_NoSum)

    # with sklearn
    precision_with = precision_score(y_test, y_pred, average='macro')
    recall_with = recall_score(y_test, y_pred, average='macro')

    # calculating f1 using recall and pricision
    f1_NoSklearn = 2 * ((precision * recall)/(precision + recall))

    # f1 with sklearn
    f1 = f1_score(y_test, y_pred, average="macro")

    

    print("KNN Accuracy with Sklearn: ", accuracy)
    print("KNN Accuracy without Sklearn: ", accuracy_NoSkLearn)
    print("KNN f1 score with: ", f1)
    print("KNN f1 score without SKlearn: ", f1_NoSklearn)
    print("KNN precision with sklearn:", precision_with)
    print("KNN precision without sklearn:", precision)
    print("KNN recall with sklearn:", recall_with)
    print("KNN recall without sklearn:", recall)

    conf_plot = plt.show()
