
from sklearn.model_selection import train_test_split    #Split the dataset into training and validation
#from utils import *
import pandas as pd
from NaiveBayesClassifier import *
import os



def main():

    # read the list of unique vocabs as features
    features =[]
    with open(os.getcwd() +'/vocabulary.txt' , 'r') as f:
        features = f.readlines()

    features = [line.rstrip('\n') for line in features]
    features.insert(0, 'ID')
    features.insert(len(features), 'Label')

    print("Reading the training data:")
    train = pd.read_csv(os.getcwd() +'/training.csv', names=features, header = None)


    #split the data into training and validation

    # This will take some time to complete. UNCOMMENT THE NEXT 4 LINES,
    # if you want to work with the validation dataset
    #X_train, X_val, y_train, y_val = train_test_split(train.loc[:, train.columns != "Label"] ,
    #                                                  train["Label"],
    #                                                  test_size=0.2,
    #                                                  random_state=1)


    #Set-up the best hyper-parameters,
    print("\nTraining started ...")
    print("Training might take a while to complete!")
    classifier = NaiveBayesClassifier(data=train, beta = 0.011)

    #train the model
    classifier.fit()

    #final accuracy score:
    y_pred = classifier.predict(train)
    print ("- Train: ", classifier.accuracy_score(y_pred, train["Label"]))



if __name__ == "__main__":
    main()
