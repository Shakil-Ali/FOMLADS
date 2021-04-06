import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def load_data():
    """
    Function that loads data from csv file to dataframe. We are using header=0 to specify that the first row is heade
    We are dropping using drop function the unnecessary columns and return the updated dataframe
    Returns
    -------
    df: DataFrame
    """
    df = pd.read_csv("data/income_evaluation.csv", sep=r'\s*,\s*',
                     header=0, encoding='ascii', engine='python', na_values="?")
    df = df.drop(['fnlwgt', 'education', 'occupation', 'capital-gain', 'capital-loss'], axis=1)

    bins = [15, 24, 55]
    names = ['15-24', '25-54', '55+']
    d = dict(enumerate(names, 1))
    df['AgeRange'] = np.vectorize(d.get)(np.digitize(df['age'], bins))
    df = df.drop(['age'], axis=1)

    return df


def reduce_dataset():
    """
    After loading the data we have to shuffle them so we can reduce the data size to 10000 instead of 32561.
    We reducing the data to improve performance.
    Returns
    -------
    df: DataFrame
    """
    df = load_data()
    df = shuffle(df, random_state=42)
    return df.iloc[22561:]


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
    reduced_dataset = reduce_dataset()
    train_portion = .6
    test_portion = .2
    return np.split(reduced_dataset,
                    [int(train_portion * len(reduced_dataset)),
                     int((train_portion + test_portion) * len(reduced_dataset))])


