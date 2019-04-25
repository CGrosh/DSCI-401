from knn import *
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from util import cat_features
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
#from scipy.spatial.distance import euclidean
data = pd.read_csv('assignment3_data/churn_data.csv')
lst_x = list(data.columns)
lst_x.remove('Churn')
lst_x.remove('CustID')
data_x = data[lst_x]
data_y = data['Churn']

# Create the Label Encoding Pipeline
le = preprocessing.LabelEncoder()

# Pass the Categorical X values through the pipeline
data_x['Gender'] = le.fit_transform(data_x['Gender'])
data_x['Income'] = le.fit_transform(data_x['Income'])

# Pass the Response variable through the pipeline
data_y = le.fit_transform(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=4)
#print(x_train.head())
print(type(x_train))

#type(x_train)

def euclidean(x1, x2):
    dist = math.sqrt(sum([(a-b)**2 for a, b in zip(x1, x2)]))
    return dist
preds = []
ks = [i for i in range(0,20)]
#for k in ks:
#    mod = knn(k, euclidean)
#    mod.fit(x_train, y_train)
#    y_hat = mod.predict(x_test)
#    preds.append(y_hat)

mod = knn(5, euclidean)
mod.fit(x_train, y_train)

y_hat = mod.predict(x_test)
print(y_hat)
#print(y_hat)
#print('--------Evaluating ModeK: k = ' + str(k) + '-------------')
print('Accuracy: ' + str(accuracy_score(y_test, y_hat)))
print('Precision: ' + str(precision_score(y_test, y_hat)))
print('Recall: ' + str(recall_score(y_test, y_hat)))
print('F1: ' + str(f1_score(y_test, y_hat)))
print('ROC AUC: ' + str(roc_auc_score(y_test, y_hat)))
print('Confusion Matrix: ' + str(confusion_matrix(y_test, y_hat)))
#print(y_train.value_counts())
