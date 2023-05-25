# Email Spam Classification using Naive Bayes
This repository contains a Python script for classifying spam and non-spam messages using the Naive Bayes algorithm with up to 98% accuracy on unseen examples. The script uses the spam_ham_dataset.csv dataset available at https://www.kaggle.com/datasets/venky73/spam-mails-dataset, which contains labeled spam and non-spam messages.

## Prerequisites
Python 3.x

NumPy

Pandas

NLTK (Natural Language Toolkit)

Matplotlib

Seaborn

Scikit-learn

## Usage
Place the spam_ham_dataset.csv file in the same directory as the Python script.


## Description
The script performs the following steps:

1. Imports the necessary libraries and packages.
2. Loads the spam_ham_dataset.csv file into a Pandas DataFrame.
3. Cleans the data by renaming the columns and removing duplicates.
4. Displays information about the DataFrame, including its shape and missing data.
5. Defines a text processing function to remove punctuation and stopwords from the messages.
6. Tokenizes and converts the text into a matrix of token counts using the CountVectorizer from Scikit-learn.
7. Splits the data into a training set (80%) and a testing set (20%).
8. Creates and trains a Naive Bayes classifier using the MultinomialNB class from Scikit-learn.
9. Evaluates the model on the training data by printing predictions, actual values, a classification report, accuracy, and a confusion matrix.
10. Displays a heatmap of the confusion matrix for the training set.
11. Evaluates the model on the testing data using similar metrics and displays a heatmap of the confusion matrix for the testing set.
## Results
The script provides insights into the performance of the Naive Bayes classifier for spam classification. It displays accuracy which reaches 98% for unseen data, precision, recall, and F1-score for both the training and testing datasets. Additionally, it visualizes the confusion matrix to show the true positive, true negative, false positive, and false negative predictions.
