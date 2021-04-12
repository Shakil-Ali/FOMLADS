import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data

from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(file):
    X_train, X_test, y_train, y_test = prepare_data(file)

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, 'Random Forest')

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_NoSkLearn = np.sum(np.equal(y_test,y_pred))/len(y_test)

    # Calculating confusion again to get precision and recall
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    # Recall and precision from confusion matrix - without sklearn
    recall_NoSum = np.diag(cm) / np.sum(cm, axis = 1)
    precision_NoSum = np.diag(cm) / np.sum(cm, axis = 0)
    precision = np.mean(precision_NoSum)
    recall = np.mean(recall_NoSum)

    # Recall and precision With sklearn
    precision_with = precision_score(y_test, y_pred, average='macro')
    recall_with = recall_score(y_test, y_pred, average='macro')

    # F1 - without sklearn
    f1_NoSklearn = 2 * ((precision * recall)/(precision + recall))

    # F1 - with sklearn
    f1 = f1_score(y_test, y_pred, average="macro")

    # Printing Evaluation Metrics
    print("Random Forest Accuracy with Sklearn: ", accuracy)
    print("Random Forest Accuracy without Sklearn: ", accuracy_NoSkLearn)
    print("Random Forest f1 score with: ", f1)
    print("Random Forest f1 score without SKlearn: ", f1_NoSklearn)
    print("Random Forest precision with sklearn:", precision_with)
    print("Random Forest precision without sklearn:", precision)
    print("Random Forest recall with sklearn:", recall_with)
    print("Random Forest recall without sklearn:", recall)

    ##################################################
    # Metrics
    # print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    # accuracy = accuracy_score(y_test, y_pred)
    # accuracy_NoSkLearn = np.sum(np.equal(y_test,y_pred))/len(y_test)
    # f1 = f1_score(y_test, y_pred, average="macro")
    # prec_score = precision_score(y_test, y_pred, average="macro")

    # print("Random Forest Accuracy with Sklearn: ", accuracy)
    # print("Random Forest Accuracy without Sklearn: ", accuracy_NoSkLearn)
    # print("Random Forest f1 score: ", f1)
    # print("Random Forest Precision Score: ", prec_score)
    ####################################################

    # Drawing Decision Tree
    #plt.figure(figsize=(5, 5))
    #     for i in range(len(clf.estimators_)):
    #             tree.plot_tree(clf.estimators_[i])
    #tree.plot_tree(clf.estimators_[99])
    #plt.show()
