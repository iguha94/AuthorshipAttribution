import pickle
import sys
from sklearn.neural_network import MLPClassifier
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing import text
from Jstylo9 import readability as extract_features

# Parameter Setting
datasetName = str(sys.argv[1])
authorsRequired = int(sys.argv[2])


def getData():
    picklesPath = '../../Data/datasetPickles/' + datasetName + "-" + str(authorsRequired) + "/"
    with open(picklesPath+'X_train.pickle', 'rb') as handle:
        X_train = pickle.load(handle)

    with open(picklesPath+'X_test.pickle', 'rb') as handle:
        X_test = pickle.load(handle)

    return (X_train, X_test)

def getAllData():
    (X_train_all, X_test_all) = getData()
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for (filePath, filename, authorId, author, inputText) in X_train_all:
        X_train.append(getFeatures(inputText))
        y_train.append(authorId)
    for (filePath, filename, authorId, author, inputText) in X_test_all:
        X_test.append(getFeatures(inputText))
        y_test.append(authorId)

    return X_train, X_test, y_train, y_test

def getFeatures(inputText):
    return extract_features.calculateFeatures(inputText)

X_train, X_test, y_train, y_test = getAllData()

if datasetName=='amt' and authorsRequired==5:
    clf = MLPClassifier(hidden_layer_sizes=(5, 7), activation='relu', learning_rate_init=0.009)
elif datasetName=='BlogsAll' and authorsRequired==5:
    clf = MLPClassifier(hidden_layer_sizes=(5, 7), activation='relu', learning_rate_init=0.009)
elif datasetName == 'amt' and authorsRequired == 10:
    clf = MLPClassifier(hidden_layer_sizes=(5, 9), activation='relu', learning_rate_init=0.009)
elif datasetName == 'BlogsAll' and authorsRequired == 10:
    clf = MLPClassifier(hidden_layer_sizes=(5, 9), activation='relu', learning_rate_init=0.009)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Starting Training...")
clf.fit(X_train, y_train)
print("Starting Saving...")

if not os.path.exists("trainedModels/" + datasetName + "-" + str(authorsRequired) + "/" ):
    os.makedirs("trainedModels/" + datasetName + "-" + str(authorsRequired) + "/")

filename = "trainedModels/" + datasetName + "-" + str(authorsRequired) + "/" + 'trained_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

predicted = clf.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, predicted))
