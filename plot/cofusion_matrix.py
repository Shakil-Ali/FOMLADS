from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_test, y_pred, name):
    con_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(con_matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = 'Confusion matrix of the '+name+' classifier'
    ax.set_title(title)
    return fig,ax
