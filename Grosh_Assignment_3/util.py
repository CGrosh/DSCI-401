import pandas as pd 
import functools 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import PolynomialFeatures


# Function to detect if columns are numerical or categorical 
def cat_features(dataframe):
    td = pd.DataFrame({'a': [1,2,3], 'b': [1.0, 2.0, 3.0]})
    return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Prints standard regression error metrics given predicted and actual values 
def print_regression_error_metrics(preds, y_test):
    print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), 
                                       median_absolute_error(y_test, preds), 
                                       r2_score(y_test, preds),
                                       explained_variance_score(y_test, preds)]))