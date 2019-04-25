from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

class KNN:
    def __init__(self, k, distance_f):
        self.k = k
        self.distance_f = distance_f

    def fit(self, x, y):
        self.y_train = y
        if isinstance(x, pd.DataFrame):
            self.x_train = x.values
        else:
            print('Test data must be a dataframe')
            print(type(x))

    def predict(self, test_x):
        dist = []
        y_hat = []
        if isinstance(test_x, pd.DataFrame):
            x_test = test_x.values
        else:
            print('Data must be a dataframe')

        for i in x_test:
            emp = []
            y_emp = []
            last = []
            for x, y in zip(self.x_train, self.y_train):
                emp.append(self.distance_f(i, x))
                y_emp.append(y)

            hats = np.column_stack((y_emp, emp))
            hats = hats[hats[:,-1].argsort()]
            hats = hats[:self.k]

            for i in hats:
                last.append(list(i))
            dist.append(last)


        y_probs = [[0,0] for row in range(len(dist))]
        for row in range(len(dist)):
            for val in dist[row]:
                pyval = val[0].item()
                if pyval == 1.0:
                    y_probs[row][0] += 1
                elif pyval == 0.0:
                    y_probs[row][1] += 1
        final_output = []
        for vals in y_probs:
            if vals[0] > vals[1]:
                final_output.append(1)
            elif vals[0] < vals[1]:
                final_output.append(0)

        return final_output
