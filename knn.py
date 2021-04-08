import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

def random_forest_classifier(df):
    X = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
            'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315',
            'proline']]
    y = df['class']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

