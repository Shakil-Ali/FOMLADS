import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def plot_confusion_matrix(cm,name):
    sn.heatmap(cm, annot=True)
    plt.title('Confusion matrix of the ' + name + ' classifier')
    plt.show()


def scatter_plot(X_lda, y, name):

    plt.title(name)
    plt.scatter(
        X_lda[:, 0],
        X_lda[:, 1],
        c=y,
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'

    )
    plt.show()
