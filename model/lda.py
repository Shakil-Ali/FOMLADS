from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from prepare_data import prepare_data
from sklearn.metrics import accuracy_score
from plot.plotting_functions import plot_confusion_matrix,scatter_plot
import matplotlib.pyplot as plt


def lda(file):
    X_train, X_test, y_train, y_test = prepare_data(file)
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    y_pred = lda_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, 'LDA')
    X_lda_train = lda_model.fit_transform(X_train, y_train)
    X_lda_test = lda_model.fit_transform(X_test, y_test)
    scatter_plot(X_lda_train,y_train,'Train_LDA')
    scatter_plot(X_lda_test,y_test, 'Test_LDA')
    scatter_plot(X_lda_test, y_pred, 'PredictLDA')
    accuracy = accuracy_score(y_test, y_pred)

    print("LDA Accuracy: ", accuracy)


