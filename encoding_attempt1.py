import pandas as pd
import numpy as np
from sklearn import preprocessing

def encode():
    #importing csv file
    df = pd.read_csv('data/income_evaluation.csv', sep=r'\s*,\s*',
                            header=0, encoding='ascii', engine='python')
    # dropping unused columns
    df = df.drop(['fnlwgt', 'education', 'occupation', 'capital-gain', 'capital-loss'], axis=1)

    # making the instance threshold (e.g if there are no instances of )
    tot_instances = df.shape[0]
    threshold = tot_instances*.005
    print('the minimum count threshold is: '+ str(threshold))

    obj_columns = list(df.select_dtypes(include=['object']).columns)
    df = df.apply(lambda x: x.mask(x.map(x.value_counts())<threshold, 'RARE')if x.name in obj_columns else x)

    df_encoded = pd.get_dummies(data=df, columns=obj_columns)
    return df_encoded.head(20)

print(encode())