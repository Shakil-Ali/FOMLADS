import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import tree

# Random Forest Classifier Function 
"""
Input: Dataframe
Output: Accuracy and confusion matrix after performing random forest classification on the dataframe
"""
def random_forest_classifier(df):
    test_df = df.sample(frac=1)

    # Assigning variables with the appropriate columns    
    X = test_df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
            'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315',
            'proline']]
    y = test_df['class']

    # normalizing without sklearn (excluding 'Class" column)
    X =(X-X.mean())/X.std()    

    # Splitting data - not using model_selection    
    train_pct_index = int(0.6 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    # Accuracy
    print('Random Forest Classification Model Accuracy: ', metrics.accuracy_score(y_test, y_pred))

    # Confusion Matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.title("Confusion Matrix - Random Forest Classification")
    plt.show()

    # Plotting final decision tree
    plt.figure(figsize=(5,5))    
    # for i in range(len(clf.estimators_)):
    #     tree.plot_tree(clf.estimators_[i])
    tree.plot_tree(clf.estimators_[99])        
    plt.show()

