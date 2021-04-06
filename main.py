import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import OrdinalEncoder

from prepare_data import load_data
from model.random_forest import random_forest_classification

def main():
    df = load_data()
    
    # printing columns
    columns = df.columns
    for col in columns:
        print(col)
    
    print("Next Test")
    print("\n")

    # printing head of df
    df1 = df.head(20)

    df2 = df1.dropna()
    print(df2.head(20))

    print('\n')

    # Number of rows
    # index = df.index
    # number_of_rows = len(index)
    # print(number_of_rows)

    print('encoding part')

    # only show 'object' columns
    obj_df = df2.select_dtypes(include=['object']).copy()
    print(obj_df.head(40))

    print('\n')
    print('encoding part')
    print('\n')

    # encoding df 
    ord_enc = OrdinalEncoder()

    obj_df["workclass_code"] = ord_enc.fit_transform(obj_df[["workclass"]])
    
    obj_df["marital-status_code"] = ord_enc.fit_transform(obj_df[["marital-status"]])

    obj_df["relationship_code"] = ord_enc.fit_transform(obj_df[["relationship"]])

    obj_df["race_code"] = ord_enc.fit_transform(obj_df[["race"]])

    obj_df["sex_code"] = ord_enc.fit_transform(obj_df[["sex"]])

    obj_df["native-country_code"] = ord_enc.fit_transform(obj_df[["native-country"]])

    obj_df["income_code"] = ord_enc.fit_transform(obj_df[["income"]])

    obj_df["AgeRange_code"] = ord_enc.fit_transform(obj_df[["AgeRange"]])

    print(obj_df.head(10))

    pen_df = obj_df.drop(['workclass', 'marital-status', 'relationship', 'sex', 'native-country', 'income', 'AgeRange'], axis=1)

    print(pen_df.head(10))

    df3 = df2.drop(['workclass', 'marital-status', 'relationship', 'sex', 'native-country', 'income', 'AgeRange'], axis=1)

    print(df3.head())

    bigdata = pd.merge(pen_df,df3, on='race')

    print(bigdata.head(30))

    final_df = bigdata.drop(['race'], axis=1)
    print(final_df.head(40))

    index = bigdata.index
    number_of_rows2 = len(index)
    print(number_of_rows2)

    random_forest_classification(final_df)


if __name__ == "__main__":
   main()
   print("File one executed when ran directly")
else:
   print("File one executed when imported")        
