# BSAN_6070_CA02_SpamDetection

## Overview
This program trains a Naive Bayes Classification Model to detect Spam Emails. The model is trained with a set of emails labelled as either Spam or Not Spam. There are 702 emails equally divided into spam and non spam category. This program creates a word dictionary with the 3000 most common words within all the emails. Then this program goes through all the files (emails) and breaks up each text document (line-by-line) and counts how many times each word shows up in the document. This is saved as the feature_matrix, and the labels_array is a binary array denoting spam or not spam. Lastly, this program trains the Guassian NB model on the training data, and tests the accuracy of the model using test data. 

## Necessary Libraries
* os (Version 3.13.1)
* Numpy (Version 1.18.0)
* Collections (import Counter)
* Pandas (Version 2.2.3)
* Scikit Learn (Version 1.6.1)
Scikit Learn (Naive Bayes section with MulitnomialNB function) (Version 1.6.1)
Scikit Learn (Metrics section with accuracy_score function) (Version 1.6.1)

## Dataset and Source Code
The dataset and source code was provided by Professor Brahma. I reused his code [specifically the make_dictionary and extract_features functions to prepare the training/test data for the Gaussian NB model. I added comments and double checked the accurary of the model. The source of this data is not given but a link is provided to download the zipped data : https://drive.google.com/file/d/1ubxrG0QYYNQcs5GDNn9DiS38Kfc7N0Ws/view?usp=sharing. 

## Installation
Open the .ipynb file in your chosen environment, ensure that the data is in the same repository/working directory and then run the code.
