import pandas as pd
import numpy as np
from sklearn import preprocessing

def encoding2():

    #importing csv file
    df = pd.read_csv('data/income_evaluation.csv', sep=r'\s*,\s*',
                                header=0, encoding='ascii', engine='python')
    # dropping unused columns
    df = df.drop(['fnlwgt', 'education', 'occupation', 'capital-gain', 'capital-loss'], axis=1)

    data_cat = df[['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'income']]
    pd.get_dummies(data_cat, dummy_na=True, drop_first=True)

    df_2 = data_cat

    ohe = OneHotEncoder(categories='auto', drop='first')
    ohe.fit(df_2.fillna('Missing'))
    ohe.get_feature_names(['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'income'])

    df_3 = ohe.transform(df_2.fillna('Missing')).toarray()
    new_dataframe = pd.DataFrame(df_3, columns=ohe.get_feature_names())
    return new_dataframe