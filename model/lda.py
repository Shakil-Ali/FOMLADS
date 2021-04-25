import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from evaluation.evaluation import calculate_cm, print_evaluation_scores
from plot.plotting_functions import plot_confusion_matrix, scatter_plot
from prepare_data import prepare_data


def lda(file):
    X_train, X_test, y_train, y_test, x_tests, y_tests = prepare_data(file)
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    accuracies = []

    for i in range(0, len(x_tests)):
        y_pred = lda_model.predict(x_tests[i])
        acc = np.sum(np.equal(y_tests[i], y_pred)) / len(y_tests[i])
        accuracies.append(acc)
    y_pred = lda_model.predict(X_test)
    cm = calculate_cm(y_test, y_pred)
    plot_confusion_matrix(cm, 'LDA')
    X_lda_train = lda_model.fit_transform(X_train, y_train)
    X_lda_test = lda_model.fit_transform(X_test, y_test)
    scatter_plot(X_lda_train, y_train, 'Trained Data LDA')
    scatter_plot(X_lda_test, y_test, 'Test Data LDA')
    scatter_plot(X_lda_test, y_pred, 'Predict Data LDA')
    print_evaluation_scores('LDA', cm, y_test, y_pred)
    print("LDA cross validation accuracies:", accuracies)
