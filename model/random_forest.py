from evaluation.evaluation import calculate_cm, print_evaluation_scores
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(file):
    X_train, X_test, y_train, y_test = prepare_data(file)

    # Fitting the data
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = calculate_cm(y_test, y_pred)
    plot_confusion_matrix(cm, 'Random Forest')
    print_evaluation_scores('Random Forest', cm, y_test, y_pred)
    ##################################################
    # Metrics
    # print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    # accuracy = accuracy_score(y_test, y_pred)
    # accuracy_NoSkLearn = np.sum(np.equal(y_test,y_pred))/len(y_test)
    # f1 = f1_score(y_test, y_pred, average="macro")
    # prec_score = precision_score(y_test, y_pred, average="macro")

    # print("Random Forest Accuracy with Sklearn: ", accuracy)
    # print("Random Forest Accuracy without Sklearn: ", accuracy_NoSkLearn)
    # print("Random Forest f1 score: ", f1)
    # print("Random Forest Precision Score: ", prec_score)
    ####################################################

    # Drawing Decision Tree
    #plt.figure(figsize=(5, 5))
    #     for i in range(len(clf.estimators_)):
    #             tree.plot_tree(clf.estimators_[i])
    #tree.plot_tree(clf.estimators_[99])
    #plt.show()
