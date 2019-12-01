# import pandas as pd
# from scipy import stats
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# import eli5
# from itertools import combinations
# from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
#
# all_data = pd.read_csv('AllFeatures.csv', index_col=False)
# req_data = all_data[
#     ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM',
#      'VERB', 'average_word_length', 'digits_percentage', 'total_words', 'Author_Id']]
#
# """
#     description:
#
#     For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation.
#     Then is takes the absolute of Z-score because the direction does not matter, only if it is below the threshold.
#     all(axis=1) ensures that for each row, all column satisfy the constraint.
#     Finally, result of this condition is used to index the dataframe.
# """
# comp_req_data = req_data[(np.abs(stats.zscore(req_data)) < 3).all(axis=1)]
# y = np.asarray(req_data["Author_Id"])
#
# authors_to_pick = list(set(y))[:100]
# print(authors_to_pick)
# req_data = comp_req_data[comp_req_data['Author_Id'].isin(authors_to_pick)]
#
# X = np.asarray(req_data.loc[:, req_data.columns != 'Author_Id'])
# y = np.asarray(req_data["Author_Id"])
#
# kf = KFold(n_splits=10, random_state=42, shuffle=False)
# # kf = kf.get_n_splits(X)
#
# scores = []
# best_svr = RandomForestClassifier()
#
# for train_index, test_index in kf.split(X):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     best_svr.fit(X_train, y_train)
#     scores.append(best_svr.score(X_test, y_test))
#
# print(print(np.mean(scores)))
# SVM test 1

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import eli5
from itertools import combinations
plt.rcParams["font.family"] = "Times New Roman"
all_data = pd.read_csv('AllFeatures.csv', index_col=False)
req_data = all_data[
    ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM',
     'VERB', 'avgWordLen', 'digits %', 'tot.words', 'Author_Id']]

"""
    description:

    For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation.
    Then is takes the absolute of Z-score because the direction does not matter, only if it is below the threshold.
    all(axis=1) ensures that for each row, all column satisfy the constraint.
    Finally, result of this condition is used to index the dataframe.
"""
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

comp_req_data = req_data[(np.abs(stats.zscore(req_data)) < 3).all(axis=1)]
y = np.asarray(req_data["Author_Id"])

options = list(combinations(list(set(y)), 2))
all_feature_weights = []
for authors_to_pick in options:
    authors_to_pick = list(authors_to_pick)
    print(authors_to_pick)
    req_data = comp_req_data[comp_req_data['Author_Id'].isin(authors_to_pick)]
    features_names = list(req_data.columns)[:-1]
    X = req_data.loc[:, req_data.columns != 'Author_Id']
    y = np.asarray(req_data["Author_Id"])

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=137, stratify=y)

    svm = LinearSVC()
    svm.fit(X_train, y_train)

    feature_weights = svm.coef_.ravel()
    print(accuracy_score(y_test, svm.predict(X_test)))
    all_feature_weights.append(feature_weights)
print(all_feature_weights)
all_feature_weights = np.asarray(all_feature_weights)
feature_weights = np.mean(all_feature_weights, axis = 0)
# features_na = list(set(features_names))
# print(features_na)
# uppd_features = []
# for i in range(len(features_na)):
#     fea = features_na[i]
#     if fea == 'average_word_length':
#         uppd_features.append('avgWordLen')
#     elif fea == 'total_words':
#         uppd_features.append('tot.Words')
#     elif fea == 'digits_percentage':
#         uppd_features.append('digits %')
#     else:
#         uppd_features.append(fea)
def plotit(name):
    x = np.arange(18)
    money = feature_weights
    plt.figure(figsize=(60, 30))
    fig, ax = plt.subplots()
    plt.bar(x, money)
    plt.xticks(x, features_names,fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(rotation=35)
    plt.ylim(-1, 1)
    plt.title(name, fontsize=30)
    fig.set_size_inches(25.5, 15.5)
    plt.savefig(name + '.png')
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()
print(feature_weights)
plotit('Feature Importance')
