import math

from matplotlib import pyplot as plt
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data
from sklearn.metrics import accuracy_score, f1_score, precision_score
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
    f1 = f1_score(y_test, y_pred, average="macro")

    print("KNN Accuracy: ", accuracy)
    print("KNN f1 score: ", f1)
    conf_plot = plt.show()
