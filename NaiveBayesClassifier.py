import numpy as np
import pandas as pd

class NaiveBayesClassifier:

    def __init__(self, data , beta):
        self.data = data
        self.beta = beta
        self.logPrior = None
        self.logLikelihood = None


    def fit(self):
        """Input:
                - train-> shape (n , |V|+2):
                    where n is the set of documents,and v is the counts for each word in the vocabulary list

                - y -> shape (n,1): where n is the set of labels associated with each document in X.

            Output:
                - logPrior : P(y) -> shape (k, 1)
                                  where P(y) is the ratio of a class to the overall number of classes
                                          i.e. (#of docs labeled y_k)/total_#docs

                -logLikelihood: P(X_i|y_k) -> shape (k, v)
                                  where P(X_i|y_k) is the probability of a word in a specific class.
            """

        totalDocCount = len(self.data["Label"]) # total number of documents(m)
        counts   = dict(self.data["Label"].value_counts()) #list of classes and their counts
        classes  = sorted(counts.keys()) #list of classes
        lengthOfVocab = len(self.data.columns) -2
        alpha = self.beta + 1

        # store the priors in sorted order to facilitate the prediction by using np.dot
        #MLE for p(y)
        Prior = np.asarray([(counts[cls]/totalDocCount) for cls in classes ]) # those should sum to one

        #sum all words that appears in a specific class
        df = self.data.groupby(by="Label", axis= "rows", sort=True).sum()  # k * n

        #count of X_i in Y_k
        c = df.drop(columns = ["ID"]).to_numpy() # shape k*n

        #total words in Y_k -> shape k*1
        t = np.sum(c, axis=1) # shape k*1

        Likelihood = ( (c.T +(alpha-1)) / (t +((alpha-1)*lengthOfVocab) ) ).T


        self.logPrior = np.log2(Prior)
        self.logLikelihood = np.log2(Likelihood)

        return self.logPrior, self.logLikelihood


    def predict(self, X_test):
        """Input:
            - X_test -> shape (n,v):
                where n is the set of documents,and v is the counts for each word in the vocabulary list.

            - logPrior : log2(P(y)) -> shape (k, 1)
                where P(y) is the ratio of a class to the overall number of classes
                                      i.e. (#of docs labeled y_k)/total_#docs

            -logLikelihood: log2(P(X_i|y_k)) -> shape (k, v)
                     where P(X_i|y_k) is the probability of a word in a specific class.
        Output:

            - predictions: y_hat -> shape(n,1)
                    set of the predicted classes.
        """

        cols = X_test.columns
        if "ID" in cols and "Label" in cols:
            examples = X_test.drop(columns=["ID" , "Label"])
        elif "Label" in cols:
            examples = X_test.drop(columns=["Label"])
        elif "ID" in cols:
            examples = X_test.drop(columns=["ID"])


        #UNCOMMENT line 84 and COMMENT line 85, for exist/ Not exist prediction.

        #result = np.dot(np.where(examples == 0, 0, 1), logLikelihood.T) + logPrior
        result = np.dot(examples, self.logLikelihood.T) + self.logPrior


        predictions = []
        for instance in result: #iterate over all examples
            argmax = -1
            maxVal = - np.inf

            for i, item in enumerate(instance): #return the max class
                if item > maxVal:
                    argmax = i+1  #i+1 because i starts from 0, but the classes start from 1
                    maxVal = item

            predictions.append(argmax)

        return predictions


    def accuracy_score(self, y_pred , y_true):
        """
            Input:
                - X_pred -> shape (n,1):
                    set of the predicted classes.

                - y_true : P(y) -> shape (n,1):
                    set of the true classes.
            Output:

                - accuracy: y_hat -> int
                        the accuracy of the predictions
        """

        return np.sum(np.equal(y_true, y_pred)) / len(y_true)
