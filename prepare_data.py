import pandas as pd
from sklearn.utils import shuffle


def prepare_data(file):
    """
    We are reading the file and creating a dataframe.
    We are shuffling using 42. We MUST have a specific number so they can reproduce our results.
    We are normalizing using --> (x-mu)/std
    :param file:
    :return X_train,X_test,y_train,y_test: All the sets that we need to run our model and find accuracy.
    """
    data = pd.read_csv(file, names=['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                    'total_phenols',
                                    'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                                    'OD280/OD315',
                                    'proline'])
    # Shuffling data using number 42 so we can reproduce the results.
    data = shuffle(data, random_state=42)
    X_non = data[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                  'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315',
                  'proline']]

    # normalizing without sklearn (excluding 'Class" column)
    X = (X_non - X_non.mean()) / X_non.std()

    y = data['class']

    # splitting data without sklearn
    train_pct_index = int(0.6 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    return X_train, X_test, y_train, y_test
