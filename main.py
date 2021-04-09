import pandas as pd

from model.random_forest import random_forest_classifier
from Visuals.data_set_visuals import density
from model.knn import knn
from prepare_data import prepare_data

def main():
    df = prepare_data('data/wine.data')
    # normalized_df=(df-df.mean())/df.std()
    # print(df.head())
    while True:
        print("****************************************")
        print("Please select an option to view model")
        print("****************************************")
        print("input [1] to view Random Forest Classifier")
        print("input [2] to view K Nearest neighbors")
        print("input [3] to view data set")
        print("input [4] to view data set statistics and graphs")
        print("****************************************")

        choice = int(input("Selection: "))
        if choice == 1:
            random_forest_classifier(df)
        elif choice == 2:
            knn(df)
        elif choice == 3:
            print(df)
        elif choice == 4:
            while True:
                print("****************************************")
                print("Choose which graph/stat to view")
                print("****************************************")
                print("input [1] to view Density Functions")
                print("input [2] to view K Nearest neighbors")
                print("input [3] to view data set")
                print("****************************************")
                stat_choice = int(input("Choice: "))
                if stat_choice == 1:
                    density()
        else: 
            print(str(choice) + " is not a valid option")

if __name__ == "__main__":
    main()