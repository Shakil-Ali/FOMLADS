import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def plot_confusion_matrix(cm, name):
    sn.heatmap(cm, annot=True)
    plt.title('Confusion matrix of the ' + name + ' classifier')
    plt.show()


def scatter_plot(X_lda, y, name):
    fig, ax = plt.subplots()
    ax.set_title(name)

    scatter =ax.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='rainbow', alpha=0.7, edgecolors='b')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

def data_stats(file):
    df = pd.read_csv(file, names=['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                  'total_phenols',
                                  'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                                  'OD280/OD315',
                                  'proline'])

    X_non = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                'flavanoids', 'non_flavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315',
                'proline']]
    X_non = X_non.round(3)
    # normalizing without sklearn (excluding 'Class" column)
    X = (X_non - X_non.mean()) / X_non.std()
    lda_model = LinearDiscriminantAnalysis()
    X_lda_whole_set = lda_model.fit_transform(X_non, df['class'])
    scatter_plot(X_lda_whole_set,df['class'], 'Data Set')
    X = X.join(df['class'])
    class_feature_means = pd.DataFrame(columns=df['class'].unique())
    print(class_feature_means)
    for c, rows in df.groupby('class'):
        rows = rows.drop('class', axis=1)
        class_feature_means[c] = rows.mean().round(2)
    print('Non standardized mean values\n', class_feature_means)
    class_feature_min = pd.DataFrame(columns=df['class'].unique())
    for c, rows in df.groupby('class'):
        rows = rows.drop('class', axis=1)
        class_feature_min[c] = rows.min()
    print('Non standardized min values\n', class_feature_min)
    class_feature_max = pd.DataFrame(columns=df['class'].unique())
    for c, rows in df.groupby('class'):
        rows = rows.drop('class', axis=1)
        class_feature_max[c] = rows.max()
    print('Non standardized max values\n', class_feature_max)
    for c, rows in df.groupby('class'):
        rows = rows.drop('class', axis=1)
        class_feature_means[c] = rows.min()
    print('Non standardized min values\n', class_feature_means)
    class_feature_standardized_means = pd.DataFrame(columns=df['class'].unique())
    for c, rows in X.groupby('class'):
        rows = rows.drop('class', axis=1)
        class_feature_standardized_means[c] = rows.mean()
    print('Standardized mean values\n', class_feature_standardized_means)
