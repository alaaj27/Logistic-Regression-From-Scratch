import numpy as np
import pandas as pd
from sklearn import preprocessing
import operator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

def preprocess (df):

    """perform the preprocessing steps:
        - remove stop words
        - perform stemming

        Input:
            df : dataframe of shape (n,|V|)
        output:
            df : preprocessed dataframe with some of the words removed such as stop words.
    """

    stp_words = nltk.corpus.stopwords.words("english")

    numOfstpWords = 0
    numOfFeatures = len (df.columns)

    for w in df.columns:
        if w == "Label" and w == "ID":
            continue
        elif w in stp_words:
            df.drop(w, axis=1, inplace=True)
            numOfstpWords += 1

    print("Number of features BEFORE stop words removal: ", numOfFeatures)
    print("Number of features AFTER stop words removal: ", len (df.columns) -2 )


    #initialize the Stemmer
    ps = nltk.stem.porter.PorterStemmer()

    vocab = [word for word in df.columns if word != "ID" and word != "Label" ]
    uniqueFeatures = {ps.stem(word):[] for word in vocab}

    for word in vocab:
        uniqueFeatures[ps.stem(word)].append(word)


    df_New = df[["ID", "Label"]].copy()

    for target , words  in uniqueFeatures.items():
        df_New[target] = df.loc[:,words].sum(axis=1)

    print("Number of features BEFORE stemming: ", len (df.columns) -2 )
    print("Number of features AFTER stemming: ", len (df_New.columns) -2)

    return df_New



def Normalization(df, Standaraziation=False, minMaxNormalize=True , MeanNormalize = True):
    """
        Given a dataframe, this function returns a normalized set of features (X) and extracts the set
        of labels (y).

        Input:
            - df -> shape (m,n+2):
                Where m is the number of documents,and n is the number of features, including Label and ID.

            - Standaraziation: if True, the data will be normalized using Z-score Normalization.

            - minMaxNormalize: if True, the data will be Rescaled using min-max normalization.

            - MeanNormalize: L1 norm mean normalization.

        Output:

            - predictions: y_hat -> shape(n,1)
                    set of the predicted classes.
        """

    columns = df.columns

    if "Label" in columns and "ID" in columns:
        y = df["Label"]
        arr = df.drop(['Label', 'ID'], axis=1).to_numpy()

    elif "Label" in columns:
        y = df["Label"]
        arr = df.drop(['Label'], axis=1).to_numpy()

    elif "ID" in columns:
        y = None
        arr = df.drop(['ID'], axis=1).to_numpy()

    else:
        y = None
        arr = df.to_numpy()

    if MeanNormalize:
        arr = preprocessing.normalize(arr, norm='l1' , axis =0)


    if Standaraziation:
        for i in range(arr.shape[1]):
            std = arr[:,i].std()

            if std == 0: #to avoid zero division
                arr[:,i] =np.divide ((arr[:,i] - arr[:,i].mean()), 1)
            else:
                arr[:,i] =np.divide ((arr[:,i] - arr[:,i].mean()), arr[:,i].std())


    if minMaxNormalize:
        for i in range(arr.shape[1]):
            den = (arr[:,i].max() - arr[:,i].min())

            if den == 0 : #to avoid zero division
                arr[:,i] = np.divide ((arr[:,i] - arr[:,i].min()), 1 )
            else:
                arr[:,i] = np.divide ((arr[:,i] - arr[:,i].min()), den)


    #X with the bias term added, y is None when we process testing data!
    return np.c_[np.ones((arr.shape[0],1)),arr] , y


def plot_confusion_matrix(y_true, y_pred , title='Confusion matrix', cmap=plt.cm.gray_r):
    """
        plot a confusion matrix of true labels and predicted labels.
    """

    confM = pd.DataFrame(confusion_matrix(y_true, y_pred))
    plt.matshow(df_confusion, cmap=cmap)


    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

#Question 6: Propose a method for ranking the words in the dataset based on how much the classifier
#‘relies on’ them when performing its classification (hint: information theory will help).
def analyze(logLikelihood, vocabList, best_n =100):

    Likelihood = 2**logLikelihood
    entropy = - np.sum(Likelihood * logLikelihood, axis=0)

    WordsWithIndx = {indx:entropy[indx] for indx in range(entropy.shape[0])}


    dict1 = dict(reversed(sorted(WordsWithIndx.items(), key=operator.itemgetter(1))))

    bestWords = []
    for i in list(dict1.keys())[:best_n]:
        bestWords.append(vocabList[i+1])

    return bestWords
