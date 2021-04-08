import pandas as pd
import numpy as np 
import math
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# im just importing from the csv file for now.  Was having trouble with dataframe but will replace it later
df = pd.read_csv('FOMLADS Name.csv')

# calling variables
df_alc = df['alcohol']
df_met = df['malic_acid']
df_ash = df['ash']
df_alcalin = df['alcalinity_of_ash']
df_mag = df['magnesium']
df_phen = df['total_phenols']
df_flav = df['flavanoids']
df_nonflav = df['nonflavanoid_phenols']
df_proan = df['proanthocyanins']
df_color = df['color_intensity']
df_hue = df['hue']
df_od = df['od280/od315_of_diluted_wines']
df_prol = df['proline']


# **************************************************
# THIS IS A FUNCTION THAT SHOWS BOX AND WHISKER PLOT BELOW
# **************************************************
def box_whisker_plot(name, input_plot):

    desc = df.describe()
    # creating figure
    fig = plt.figure(1, figsize=(5, 5))

    # alcohol plot
    ax = fig.add_subplot(111)
    ax.title.set_text(name)

    bp = ax.boxplot(input_plot)

    # uncomment below to save the plots for our report
    # fig.savefig('BoxWhiskerPlot.png', bbox_inches='tight')

    return desc, plt.show()

# uncomment the below line and fill in with title you want and  argument from the list (line 15-27); there is an example already filled in
#print(box_whisker_plot('alcohol', df_alc))



# **************************************************
# THIS IS A FUNCTION THAT SHOWS THE DENSITIES BELOW
# **************************************************
def density(plot):
    plt.style.use('dark_background')
    df = pd.read_csv('FOMLADS Name.csv')
    sns.distplot(plot, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    return plt.show()

# uncomment the below line and fill in with argument from the list (line 15-27) ; there is an example already filled in
#print(density(df_alc))


