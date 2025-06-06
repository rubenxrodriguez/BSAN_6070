# BSAN6070_CA03_Decision_Tree

## Overview
This is an income classifier using a decision tree model. This model will predict whether individuals' income is greater or lower than 50k per year. The data comes from the Census Bureau, and has columns such as hours worked per week, education, race and sex, and age. The provided code first inspects the data to see if any cleaning is necessary, then it splits the data into training and testing splits. Next, we train a decision tree on the data, and we optimize the hyper-parameters using a manual grid search method. Lastly, we inspect our model by visualizing the decision tree, seeing how fast our model is trained, and looking at prediction probabilities.

## Necessary Libraries
* os (Version 3.13.1)
* Numpy (Version 1.18.0)
* Pandas (Version 2.2.3)
* Scikit Learn (Version 1.6.1)
* Scikit Learn (DecisionTreeClassifier function from sklearn.tree) (Version 1.6.1)
* Scikit Learn (Metrics section with accuracy_score, precision_score, recall_score, f1_score, and confusion_matrix function) (Version 1.6.1)
* Scikit Learn (export_graphviz function from sklearn.tree) (Version 1.6.1)
* Time (Version 3.13.2)
* Graphviz (Version 0.20.3)

## Dataset and Source Code
The dataset was provided by Professor Brahma. The link to the data source is : https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true. 

## Installation
Open the .ipynb file in your chosen environment, ensure that the data is in the same repository/working directory and that you have installed the necessary libraries, then run the Decision_Tree_Algorithm.ipynb file.
