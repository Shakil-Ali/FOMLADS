import math

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def k_value_graph(X_train, y_train, X_test, y_test):
    # getting k
    k = math.ceil((math.sqrt(len(y_test))))

    # ks below
    kmin1 = k - 8

    # ks above
    kplus = k + 12
    kplus1 = k + 32
    kplus2 = k + 62
    kplus3 = k + 92

    # classify
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred = calculate_y_pred(k, X_test, X_train, y_train)
    # classify 8 below k
    y_predmin1 = calculate_y_pred(kmin1, X_test, X_train, y_train)
    # classify 11 above k
    y_predplus = calculate_y_pred(kplus, X_test, X_train, y_train)
    # classify 31 above k
    y_predplus1 = calculate_y_pred(kplus1, X_test, X_train, y_train)
    # classify 61 above k
    y_predplus2 = calculate_y_pred(kplus2, X_test, X_train, y_train)
    # classify 91 above k
    y_predplus3 = calculate_y_pred(kplus3, X_test, X_train, y_train)

    # accuracies
    accuracy = accuracy_score(y_test, y_pred)
    accuracymin1 = accuracy_score(y_test, y_predmin1)
    accuracyplus = accuracy_score(y_test, y_predplus)
    accuracyplus1 = accuracy_score(y_test, y_predplus1)
    accuracyplus2 = accuracy_score(y_test, y_predplus2)
    accuracyplus3 = accuracy_score(y_test, y_predplus3)

    # creating x and y axis; singling out the k value used in the model
    x = [1, 21, 41, 71, 101]
    x1 = [9]
    y = [accuracymin1, accuracyplus, accuracyplus1, accuracyplus2, accuracyplus3]
    y1 = [accuracy]

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    # plotting
    ax.scatter(x, y, marker="o", color="red")
    ax.plot(x, y, 'yo')
    ax.plot(x1, y1, 'ro', label='Optimal = sqrt(len(N))')
    fig.suptitle('Different K-value Accuracies', fontsize=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("K-value")
    ax.set_xticklabels([1, 9, 20, 40, 70, 100])

    ax.legend()

    # uncomment below to save the figure as png
    # plt.savefig('knn_accuracies.png', bbox_inches='tight')

    plt.show()


def calculate_y_pred(k, X_test, X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)
