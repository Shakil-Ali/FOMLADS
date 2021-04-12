import numpy as np
import pandas as pd


def calculate_cm(y_test, y_pred):
    return pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])


def accuracy(y_test, y_pred):
    return np.sum(np.equal(y_test, y_pred)) / len(y_test)


def recall(cm):
    recall_NoSum = np.diag(cm) / np.sum(cm, axis=1)
    return np.mean(np.mean(recall_NoSum))


def precision(cm):
    precision_NoSum = np.diag(cm) / np.sum(cm, axis=0)
    return np.mean(precision_NoSum)


def f1_recall_precision(cm):
    prec = precision(cm)
    rec = recall(cm)
    f1_NoSklearn = 2 * ((prec * rec) / (prec + rec))
    return f1_NoSklearn, rec, prec


def print_evaluation_scores(name, cm, y_test, y_pred):
    f1, rec, prec = f1_recall_precision(cm)
    acc = accuracy(y_test, y_pred)
    print(f"{name} Accuracy : {acc:0.4f}\n"
          f"{name} f1_score: {f1:0.4f}\n"
          f"{name} Recall: {rec:0.4f}\n"
          f"{name} Precision: {prec:0.4f}\n")
