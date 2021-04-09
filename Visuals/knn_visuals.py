import pandas as pd
import numpy as np 
import math

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def k_value_graph():
    df_pre = pd.read_csv('FOMLADS Name.csv')
    df = df_pre.sample(frac=1)

    # seperating columns 

    # normalizing without sklearn (excluding 'Class" column)
    X_non = df.iloc[:,0:14]
    X =(X_non-X_non.mean())/X_non.std()    

    y = df.iloc[:,14]

    # splitting data without sklearn
    train_pct_index = int(0.6 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

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
    classifier = KNeighborsClassifier(n_neighbors=k,p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # classify 8 below k
    classifiermin1 = KNeighborsClassifier(n_neighbors=kmin1,p=2, metric='euclidean')
    classifiermin1.fit(X_train, y_train)
    y_predmin1 = classifiermin1.predict(X_test)

    # classify 11 above k
    classifierplus = KNeighborsClassifier(n_neighbors=kplus,p=2, metric='euclidean')
    classifierplus.fit(X_train, y_train)
    y_predplus = classifierplus.predict(X_test)

    # classify 31 above k
    classifierplus1 = KNeighborsClassifier(n_neighbors=kplus1,p=2, metric='euclidean')
    classifierplus1.fit(X_train, y_train)
    y_predplus1 = classifierplus1.predict(X_test)

    # classify 61 above k
    classifierplus2 = KNeighborsClassifier(n_neighbors=kplus2,p=2, metric='euclidean')
    classifierplus2.fit(X_train, y_train)
    y_predplus2 = classifierplus2.predict(X_test)

    # classify 91 above k
    classifierplus3 = KNeighborsClassifier(n_neighbors=kplus3,p=2, metric='euclidean')
    classifierplus3.fit(X_train, y_train)
    y_predplus3 = classifierplus3.predict(X_test)

    # accuracies
    accuracy = accuracy_score(y_test, y_pred)
    accuracymin1 = accuracy_score(y_test, y_predmin1)
    accuracyplus = accuracy_score(y_test, y_predplus)
    accuracyplus1 = accuracy_score(y_test, y_predplus1)
    accuracyplus2 = accuracy_score(y_test, y_predplus2)
    accuracyplus3 = accuracy_score(y_test, y_predplus3)

    # creating x and y axis; singling out the k value used in the model
    x = [1,21, 41, 71, 101]
    x1 = [9]
    y = [accuracymin1, accuracyplus, accuracyplus1, accuracyplus2, accuracyplus3]
    y1 = [accuracy]

    plt.style.use('dark_background')

    # plotting
    plt.scatter(x, y, marker="o", color="red")

    plt.plot(x, y, 'yo')
    plt.plot(x1, y1, 'ro')

    plt.title("Different K-value Accuracies")
    plt.xlabel('K-value')
    plt.ylabel("Accuracy (%)")
    plt.xticks([1,9,20, 40,70,100])
    plt.style.use('dark_background')
    # uncomment below to save the figure as png
    #plt.savefig('knn_accuracies.png', bbox_inches='tight')
    
    plt.show()

print(k_value_graph())


