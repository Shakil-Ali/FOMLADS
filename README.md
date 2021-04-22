# FOMLADS

## Table of contents
* [General info](#general-info)
* [Libraries](#libraries)
* [Structure](#structure)
* [Setup](#setup)
* [Running Code](#running-code)

## General info
This is the repository for Group 1's Assessment Code (Foundations of Machine Learning and Data Science).
	
## Libraries
The required libraries to run the code:
* scikit-learn
* seaborn
* numpy
* matplotlib
* pd

## Structure
The structure of the repository is as follows:
* data - this directory contains the (wine) dataset file - wine.data
* evaluation - this directory contains the evaluation file (metrics: confusion matrix, accuracy, f1_score, recall, precision) - evaluation.py 
* model - this directory contains the classification models (Random Forest, KNN, LDA, Logistic Regression) python files  - knn.py, lda.py, log_regression.py, random_forest.py
* plot - this directory contains files for the plots of the models - data_set_visuals.py, knn_visuals.py, plotting_functions.py
* main.py - this is the main file containing the options for running the multiple models
* prepare_data.py - this is the data preparation file which prepares the dataset into test and training

## Setup
To begin running this project, 'cd' to repository first:

```
$ cd FOMLADS
```

## Running Code

To run classification using all the models at the same time, run the following command:
```
$ python main.py data/wine.data all
```

To run classification using just 'Random Forest', run the following command:
```
$ python main.py data/wine.data random_forest
```

To run classification using just 'KNN', run the following command:
```
$ python main.py data/wine.data knn
```

To run classification using just 'LDA', run the following command:
```
$ python main.py data/wine.data lda
```

To run classification using just 'Logistic Regression', run the following command:
```
$ python main.py data/wine.data log_reg
```

