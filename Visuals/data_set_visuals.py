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

# THESE ARE THE VARIABLES THAT YOU WILL HAVE TO USE IN THE BELOW FUNCTIONS
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
#print(box_whisker_plot('nonflavanoid_phenols', df_nonflav))



# **************************************************
# THIS IS A FUNCTION THAT SHOWS THE DENSITIES BELOW
# **************************************************
def density():
    plt.style.use('dark_background')
    df = pd.read_csv('FOMLADS Name.csv')
    print("choose a density function")
    print("****************************************")
    print("input [1] to view Alcohol")
    print("input [2] to view malic_acid")
    print("input [3] to view ash")
    print("input [4] to view alcalinity_of_ash")
    print("input [5] to view magnesium")
    print("input [6] to view total_phenols")
    print("input [7] to view flavanoids")
    print("input [8] to view nonflavanoid_phenols")
    print("input [9] to view proanthocyanins")
    print("input [10] to view color_intensity")
    print("input [11] to view hue")
    print("input [12] to view od280/od315_of_diluted_wines")
    print("input [13] to view proline")
    print("****************************************")
    choice = int(input("Choice: "))
    print("****************************************")
    if choice == 1:
        sns.distplot(df_alc, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 2:
        sns.distplot(df_met, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 3:
        sns.distplot(df_ash, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 4:
        sns.distplot(df_alcalin, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 5:
        sns.distplot(df_mag, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 6:
        sns.distplot(df_phen, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 7:
        sns.distplot(df_flav, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 8:
        sns.distplot(df_nonflav, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 9:
        sns.distplot(df_proan, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 10:
        sns.distplot(df_color, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 11:
        sns.distplot(df_hue, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 12:
        sns.distplot(df_od, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    elif choice == 13:
        sns.distplot(df_prol, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
    return plt.show()



