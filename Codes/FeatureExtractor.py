import os
import writeprintsStatic as ws
import io
import pickle
import pandas as pd

def load_features(file_path):
    with open(file_path, 'rb') as f:
        mynewlist = pickle.load(f)
        return mynewlist

def save_features(file_path, listt):
    with open(file_path, 'wb') as f:
        pickle.dump(listt, f)

def load_feature_vector(feature_path):
    features = load_features(feature_path)
    return [x[1] for x in features]

def extract_features():
    all_documents = []
    for root, dirs, files in os.walk("BlogsAll/"):
        for file in files:
            if file.endswith(".txt"):
                all_documents.append(os.path.join(root, file))

    all_author_names = []
    all_document_names = []
    all_document_paths = []
    all_document_feature_paths = []

    counter = 1
    for document in all_documents:
        print(document, '===', 'Processing Document: ', counter, '/', len(all_documents))
        counter+=1
        document_name = document.split('/')[-1]
        author_name = document.split('/')[-2]
        document_path = document

        inputText = io.open(document_path, "r", errors="ignore").readlines()
        inputText = ''.join(str(e) + "" for e in inputText)

        features = ws.calculateFeatures(inputText)
        featue_path = '/'.join(document_path.split('/')[:-1]) + '/' + document_name.split('.')[0] + '.pkl'

        save_features(featue_path, features)

        all_author_names.append(author_name)
        all_document_names.append(document_name)
        all_document_paths.append(document_path)
        all_document_feature_paths.append(featue_path)


    df = pd.DataFrame({
        'author_name': all_author_names,
        'document_name': all_document_names,
        'document_path': all_document_paths,
        'feature_path': all_document_feature_paths
    })
    df.to_csv('all_data.csv', index=False)

## load features with tuples (feaure_name, feature_value)
# print(load_features('BlogsAll/263212/14-263212.pkl'))

## load features vectors for classifcation
# print(load_feature_vector('BlogsAll/263212/14-263212.pkl'))