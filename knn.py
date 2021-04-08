import pandas as pd
import numpy as np 
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def knnWithSklearn():
    df = pd.read_csv('FOMLADS Name.csv')

    # seperating row
    X = df.iloc[:,0:14]
    y = df.iloc[:,14]

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.4)

    # scaling data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # getting k
    k = math.ceil((math.sqrt(len(y_test))))

    #classify
    classifier = KNeighborsClassifier(n_neighbors=k,p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return y_pred, cm, accuracy

def knnWithOut():





