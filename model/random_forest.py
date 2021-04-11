import numpy as np
import matplotlib.pyplot as plt

from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data

from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(file):
    X_train, X_test, y_train, y_test = prepare_data(file)

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, 'Random Forest')

    # Metrics
    # print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_NoSkLearn = np.sum(np.equal(y_test,y_pred))/len(y_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec_score = precision_score(y_test, y_pred, average="macro")

    print("Random Forest Accuracy with Sklearn: ", accuracy)
    print("Random Forest Accuracy without Sklearn: ", accuracy_NoSkLearn)
    print("Random Forest f1 score: ", f1)
    print("Random Forest Precision Score: ", prec_score)

    # Drawing Decision Tree
    #plt.figure(figsize=(5, 5))
    #     for i in range(len(clf.estimators_)):
    #             tree.plot_tree(clf.estimators_[i])
    #tree.plot_tree(clf.estimators_[99])
    #plt.show()
