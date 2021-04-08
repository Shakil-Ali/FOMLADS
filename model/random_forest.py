import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import tree


def random_forest_classifier(df):
    test_df = df.sample(frac=1)

    # Setting variable with columns    
    X = test_df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
            'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315',
            'proline']]
    y = test_df['class']

    # Splitting data - using model_selection    
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

    # Splitting data - not using model_selection    
    train_pct_index = int(0.6 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    # Scaling data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    # Confusion Matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    # Accuracy
    print('Model Accuracy: ',metrics.accuracy_score(y_test, y_pred))
    plt.show()

    plt.figure(figsize=(5,5))    
#     for i in range(len(clf.estimators_)):
#             tree.plot_tree(clf.estimators_[i])
    tree.plot_tree(clf.estimators_[99])        
    plt.show()

