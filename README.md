# FOMLADS

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Structure](#structure)
* [Setup](#setup)
* [Running Code](#running-code)

## General info
This is the repository for Group 1's Assessment Code (Foundations of Machine Learning and Data Science).
	
## Technologies
Project is created with:
* Python version: 3
	
## Libraries
The required libraries to run the code:
* scikit-learn
* seaborn
* numpy
* matplotlib
* pd

## Structure
The structure of the repository is as follows:
* Models - directory containing the classification models (Random Forest, KNN, LDA, Logistic Regression) python files 
* Evaluation - directory 
* Plots - directory

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

