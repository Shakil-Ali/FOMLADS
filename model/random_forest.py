import matplotlib.pyplot as plt
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data
from sklearn import metrics
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
    print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred))

    #plt.figure(figsize=(5, 5))
    #     for i in range(len(clf.estimators_)):
    #             tree.plot_tree(clf.estimators_[i])
    #tree.plot_tree(clf.estimators_[99])
    #plt.show()
