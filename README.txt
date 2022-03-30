Team : RuntimeTerror
Members: Julie Hayes and Ala Jararweh
Title: Document Classification using Naive Bayes and Logistic Regression

1. Important Notes:
	- I will provide a Jupyter notebook, if you prefer working on notebooks.
	- I will provide separate files if you prefer working on the command line.
	- test.csv and train.csv will not be submitted. You need to add them manually.


2. Dependencies :
	install numpy  	— For math functions.
	install pandas 	— For using DataFrame functionality.
	install sklearn  — We use model_selection.train_test_split to split the dataset
                    into training and validation. We also use sklearn.preprocessing.normalize.
	install collections.Counter
  install operator   - used for dictionary operations.
  install matplotlib.pyplot
  install sklearn.metrics.confusion_matrix
  install NLTK      - Used for preprocessing



3. How to run (same steps for both Logistic Regression and Naive Bayes) :

	1st step: store the dataset in the same directory where main.py exists.
	2nd step: run python main.py


4. File Descriptions:

	- main.py: runs the classifier using training data.
	- utils.py: contains some python functions to facilitate data cleaning and feature extraction.
	- NaiveBayesClassifier.py: contains Naive Bayes code.
	- LogisticRegressionClassifier.py : contain Logistic Regressions code.


5. Functions Description:

a) NaiveBayesClassifier class:
	- fit : start the training process.
	- predict: predict a set of rows at once.
  - accuracy_score : calculate the accuracy.



b) LogisticRegressionClassifier class:
	- fit: start the training process
 	- predict: predict a set of rows at once.
  - accuracy_score : calculate the accuracy.

c) Utils file:
  - Normalization: Given a dataframe, this function returns a normalized set of features (X) and extracts the set
      of labels (y).
  - preprocess (df):perform the preprocessing steps:
          - stop words Removal
          - Stemming

	- plot_confusion_matrix : plot a confusion matrix of true labels and predicted labels.
  - analyze : This function answers question 6 and 7 in the report!
