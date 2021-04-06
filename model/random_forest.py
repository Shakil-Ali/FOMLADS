import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


def random_forest_classification(df):
    X = df[['marital-status_code','relationship_code','race_code', 'sex_code', 'native-country_code', 'education-num', 'hours-per-week']]
    y = df['AgeRange_code']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)

    print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
    plt.show()

