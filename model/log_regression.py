import math

import pandas as pd
from evaluation.evaluation import print_evaluation_scores, calculate_cm
from plot.knn_visuals import k_value_graph
from plot.plotting_functions import plot_confusion_matrix
from prepare_data import prepare_data

def log_reg(file):
    #code goes here
    X_train, X_test, y_train, y_test = prepare_data(file)