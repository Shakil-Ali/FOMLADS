from prepare_data import *

from sklearn.neighbors import KNeighborsClassifier

import math

"""
KNN does not work well with categorical data.  It can technically be done but ideally only with binary categories
 or categories that have a clear hierarchy (e.g. small, medium, large).  Categorical data such as workclass, country,
 race, marital status, etc confuse me in terms of preprocessing logic; I would need to convert to ONE HOT ENCODING or
 risk poor methods and do simple INTEGER ENCODING.  Would appreciate advice.   
"""

#checking category types
def cateogires():
    reduced_dataset = reduce_dataset()
    reduced_types = reduced_dataset.dtypes
    return reduced_types

# declaring variables for classefier
fulltrain_data = 6000
group_num = 3

#function for calculating k value by taking square root of rows in training set and rounding down.
def find_k(train_data):
    k = round(math.sqrt((train_data)))
    return k

# using k function for n_neighbors, number of age range groups (3) for p, and euclidean to measure distance
classifier = KNeighborsClassifier(n_neighbors=find_k(fulltrain_data), p=group_num, metric='euclidean')

print(cateogires())

