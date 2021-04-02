import pandas as pd


def load_data():
    df = pd.read_csv("data/income_evaluation.csv", sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
    df = df.drop(['fnlwgt','education','occupation','capital-gain','capital-loss'], axis=1)
    return df




