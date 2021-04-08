import pandas as pd

from model.random_forest import random_forest_classifier
from prepare_data import prepare_data

def main():
    df = prepare_data('data/wine.data')
    # print(df.head())
    random_forest_classifier(df)
    knn(df)

if __name__ == "__main__":
    main()