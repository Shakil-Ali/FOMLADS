from model.random_forest import random_forest_classifier
from plot.data_set_visuals import print_density
from model.knn import knn
from model.lda import lda
from model.log_regression import log_reg
import sys

"""SOS!!!! not so SOS now that I have your attention please check that I created a folder plot
inside the plot folder I have the function to plotting the confusion matrix 
Please move everything tha we have inside on folder Visual inside to plot folder.

"""
def main(dataset, model):
    """
    We have to option to run this function in two ways:
    1. We specify the name of the model and we will run only for this one.
    2. We ran all the models.
    :param dataset: the path of the dataset:
    :param model: name of the model we gonna use(knn, lda, etc.)
    """

    if model == 'random_forest':
        random_forest_classifier(dataset)
    if model == 'knn':
        knn(dataset)
    if model == 'lda':
        lda(dataset)
    if model == 'log_reg':
        log_reg(dataset)
    if model == 'all':
        lda(dataset)
        knn(dataset)
        random_forest_classifier(dataset)
    if model == 'density':
        print_density(dataset)



if __name__ == "__main__":
    # Using sys we can get arguments from command line
    if len(sys.argv) <= 2:
        print("Please provide the correct arguments. python main.py <dataset> <model name> ")
        sys.exit(1)
    arguments = len(sys.argv) - 1
    main(dataset=sys.argv[1], model=sys.argv[2])
