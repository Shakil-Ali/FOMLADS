import pandas as pd

from model.random_forest import random_forest_classifier
from model.knn import knn
from prepare_data import prepare_data

def main():
    df = prepare_data('data/wine.data')
    # normalized_df=(df-df.mean())/df.std()
    # print(df.head())
    random_forest_classifier(df)
    knn(df)

if __name__ == "__main__":
    main()