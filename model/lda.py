from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from prepare_data import prepare_data
from sklearn.metrics import accuracy_score
from plot.cofusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt


def lda(file):
    X_train, X_test, y_train, y_test = prepare_data(file)
    lda_model = LinearDiscriminantAnalysis()

    lda_model.fit(X_train, y_train)
    y_pred = lda_model.predict(X_test)
    fig, ax = plot_confusion_matrix(y_test, y_pred, 'LDA')
    # accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: ", accuracy)
    plt.show()

