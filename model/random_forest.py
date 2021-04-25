import numpy as np
from sklearn.ensemble import RandomForestClassifier

from evaluation.evaluation import calculate_cm, print_evaluation_scores
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data


def random_forest_classifier(file):
    X_train, X_test, y_train, y_test, x_tests, y_tests = prepare_data(file)

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracies = []
    for i in range(0, len(x_tests)):
        y_pred = clf.predict((x_tests[i]))
        acc = np.sum(np.equal(y_tests[i], y_pred)) / len(y_tests[i])
        accuracies.append(acc)
    y_pred = clf.predict(X_test)
    cm = calculate_cm(y_test, y_pred)
    plot_confusion_matrix(cm, 'Random Forest')
    print_evaluation_scores('Random Forest', cm, y_test, y_pred)
    print("Random Forest cross validation accuracies:", accuracies)
