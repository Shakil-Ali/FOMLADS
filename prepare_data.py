from sklearn.datasets import load_wine
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


def prepare_data():
    raw_data = load_wine()

    features = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    data = features
    data['target'] = raw_data['target']
    data['class'] = data['target'].map(lambda ind: raw_data['target_names'][ind])
    return data


def divide_dataset():
    """
    After reduce and shuffle the dataset we have to divide to three datasets: train_portion, valid_portion, test_portion
    train_portion = 0.6 of dataset
    valid_portion = 0.2 of dataset
    test_portion = 0.2 of dataset
    We reducing the data to improve performance.
    Returns
    -------
    3 dataframes: train,validate, test

    """
    df = prepare_data()
    df = shuffle(df, random_state=42)
    train_portion = .6
    test_portion = .2
    return np.split(df,
                    [int(train_portion * len(df)),
                     int((train_portion + test_portion) * len(df))])


print(prepare_data())

print(divide_dataset())
