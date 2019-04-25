from knn import *
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split


def euclidean(x1, x2):
    dist = math.sqrt(sum([(a-b)**2 for a, b in zip(x1, x2)]))
    return dist

samp_train = pd.DataFrame({'X1': [7, 7, 3, 1], 'X2': [7, 4, 4, 4], 'Y': [0, 0, 1, 1]})
lst = list(samp_train.columns)
lst.remove('Y')
samp_train_x = samp_train[lst]

samp_train_y = samp_train['Y']

samp_test = pd.DataFrame({'X1':[3, 4], 'X2':[7, 6]})

x = knn(3, euclidean)
x.fit(samp_train_x, samp_train_y)
print(x.predict(samp_test))
