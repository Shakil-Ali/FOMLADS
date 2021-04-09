import pandas as pd

from model.random_forest import random_forest_classifier
from model.knn import knn
from prepare_data import prepare_data

def main():
    df = prepare_data('data/wine.data')
    # normalized_df=(df-df.mean())/df.std()
    # print(df.head())
    print("****************************************")
    print("Please select an option to view model")
    print("****************************************")
    print("input [1] to view Random Forest Classifier")
    print("input [2] to view K Nearest neighbors")
    print("input [3] to view data set")
    print("****************************************")

    choice = int(input("Make a selection: "))
    if choice == 1:
        random_forest_classifier(df)
    elif choice == 2:
        knn(df)
    elif choice == 3:
        print(df)
    else: 
        print("not a valid option")

if __name__ == "__main__":
    main()