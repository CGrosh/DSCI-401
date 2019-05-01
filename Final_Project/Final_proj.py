import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import time 

data = pd.read_csv('kickstarter_projects_data/ks-projects-201612.csv', 
                   delimiter = ',', encoding='ISO-8859-1')

del data['Unnamed: 13']
del data['Unnamed: 14']
del data['Unnamed: 15']
del data['Unnamed: 16']
del data['currency ']
del data['category ']
del data['name ']
del data['ID ']
del data['usd pledged ']

diff = []
launch = []
dead = []
for i in range(len(data)):
    launch.append(data['launched '][i].split(' ')[0])
    dead.append(data['deadline '][i].split(' ')[0])
launch = pd.Series(launch)
dead = pd.Series(dead)

launch = pd.to_datetime(launch, format='%m/%d/%y', errors='coerce')
dead = pd.to_datetime(dead, format='%m/%d/%y', errors='coerce')
data['launch'] = launch
data['dead'] = dead


for i in range(len(data)):
    diff.append((data['dead'][i] - data['launch'][i]).days)

diff = pd.Series(diff)
data['Length'] = diff

check = ['failed', 'successful', 'canceled', 'live', 'undefined', 'suspended']
count = []
for i in range(len(data)):
    if data['state '][i].isdigit() == True:
        count.append(i)
    elif data['state '][i] not in check:
        count.append(i)
data = data.drop(data.index[count])
data = data.reset_index(drop=True)

data['goal '] = data['goal '].astype('float32')
data['pledged '] = data['pledged '].astype('float32')
data['Diff'] = data['goal ']-data['pledged ']
data['backers '] = data['backers '].astype('int64')

data['goal '] = data['goal '].astype('int64')
data['pledged '] = data['pledged '].astype('int64')
print("Preparing to re-Format")
prep=[]

for i in range(len(data)):
  if data['Diff'][i] <= 0:
    prep.append('successful')
  else: 
    prep.append('failure')
    
data['result'] = pd.Series(prep)

data = data[data['Length'] < 3000.0]

lst = list(data.columns)
lst.remove('launched ')
lst.remove('deadline ')
lst.remove('state ')
lst.remove('dead')
lst.remove('launch')
lst.remove('Diff')
lst.remove('result')
lst.remove('pledged ')

def cat_features(dataframe):
    td = pd.DataFrame({'a': [1,2,3], 'b': [1.0, 2.0, 3.0]})
    return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, 
                                          td['b'].dtype]), list(dataframe))
    
print(list(cat_features(data[lst])))
data_x = data[lst]
data_x = pd.get_dummies(data_x, columns=list(cat_features(data_x[lst])))
data_y = data['result']

#%%
le = preprocessing.LabelEncoder()
imp = preprocessing.Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.StandardScaler()

data_y = le.fit_transform(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                                    test_size=0.2, random_state=4)

train_x_pp = imp.fit_transform(x_train)
train_x_pp = scaler.fit_transform(train_x_pp)

test_x_pp = imp.fit_transform(x_test)
test_x_pp = scaler.fit_transform(test_x_pp)


#%%

#n_ests = [5,10,50,100]
#depths = [2,4,6,8]
#for n in n_ests:
#    for dp in depths:
#        mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
#        mod.fit(train_x_pp, y_train)
#        y_hat = mod.predict(test_x_pp)
#        print('-----Evaluating Model: n_estimators = ' + str(n) + ', max_depth = ' + str(dp) + ' ------')
#        print(accuracy_score(y_hat, y_test))
#        print(f1_score(y_hat, y_test))

mod = ensemble.RandomForestClassifier(n_estimators=50, max_depth=8)
start = time.time()
mod.fit(train_x_pp, y_train)
y_hat = mod.predict(test_x_pp)
end = time.time()
print("Training and Predicting took approximatly: " + str(end-start))
print("------Random Forest Model With n_estimators = 50, and max_depth = 8---------")
print("Accuracy Score: " + str(accuracy_score(y_hat, y_test)))
print("F1 Score: " + str(f1_score(y_hat, y_test)))

start = time.time()
k_fold = KFold(n_splits=5, shuffle=True, random_state=4) # This is 5-fold CV
k_fold_scores = cross_val_score(mod, data_x, data_y, scoring='f1_macro', cv=k_fold)
print('CV Score (F1 Macro for K-Fold): ' + str(k_fold_scores))
end = time.time()
print("K-Fold Training and Predicting took approximatly: " + str(end - start))

#%%

#log_mod = LogisticRegression()
#nb_mod = naive_bayes.GaussianNB()
#r_mod = ensemble.RandomForestClassifier(n_estimators = 50, max_depth=8)
#
#
#voting_clf = ensemble.VotingClassifier(
#        estimators=[('lr', log_mod), ('rf', r_mod), ('nb', nb_mod)], 
#        voting = 'hard')
#start = time.time()
#voting_clf.fit(train_x_pp, y_train)
#end = time.time()
#print("Voting Classifier Training time: " + str(end - start))
#start= time.time()
#y_hat = voting_clf.predict(test_x_pp)
#end = time.time()
#print("Voting Classifier Prediction Time: " + str(end - start))
#print("Voting Classifier: " + str(accuracy_score(y_hat, y_test)))



#%% 

val_data = pd.read_csv('kickstarter_projects_data/ks-projects-201801.csv', 
                   delimiter = ',', encoding='ISO-8859-1')
print(list(val_data.columns))
# Delete unnecessary columns 
#del val_data['Unnamed: 13']
#del val_data['Unnamed: 14']
#del val_data['Unnamed: 15']
#del val_data['Unnamed: 16']
del val_data['currency']
del val_data['category']
del val_data['name']
del val_data['ID']
del val_data['usd_pledged_real']
del val_data['usd_goal_real']
del val_data['usd pledged']
val_data.launched

#%%

# Segement the Date string from the Launch and Deadline columns 
diff = []
launch = []
dead = []
for i in range(len(val_data)):
    launch.append(val_data['launched'][i].split(' ')[0])
    dead.append(val_data['deadline'][i])
launch = pd.Series(launch)
dead = pd.Series(dead)
#print(dead)
# Convert the the Launch and deadline columns into datetime objects 
launch = pd.to_datetime(launch, format='%Y/%m/%d', errors='coerce')
dead = pd.to_datetime(dead, format='%Y/%m/%d', errors='coerce')
val_data['launch'] = launch
val_data['dead'] = dead

# Loop through and get the total number of days that the project was live on Kickstarter 
for i in range(len(val_data)):
    diff.append((val_data['dead'][i] - val_data['launch'][i]).days)
    
# Create a new column of the number of days 
diff = pd.Series(diff)
val_data['Length'] = diff
val_data['Length']

#%%

# Check if the row is not aligned with the columns and drop those indexes 
check = ['failed', 'successful', 'canceled', 'live', 'undefined', 'suspended']
count = []
for i in range(len(val_data)):
    if val_data['state'][i].isdigit() == True:
        count.append(i)
    elif val_data['state'][i] not in check:
        count.append(i)
if len(count) > 0:
    val_data = val_data.drop(val_data.index[count])
    val_data = val_data.reset_index(drop=True)

# Check if the amount donated is larger than the amount requested and then label it 
# as either successful or failed
val_data['goal '] = val_data['goal'].astype('float32')
val_data['pledged '] = val_data['pledged'].astype('float32')
val_data['Diff'] = val_data['goal']-val_data['pledged']
val_data['backers '] = val_data['backers'].astype('int64')

val_data['goal '] = val_data['goal '].astype('int64')
val_data['pledged '] = val_data['pledged '].astype('int64')
print("Preparing to re-Format")

prep=[]
for i in range(len(val_data)):
  if val_data['Diff'][i] <= 0:
    prep.append('successful')
  else: 
    prep.append('failure')

val_data['result'] = pd.Series(prep)
val_data['main_category '] = val_data['main_category']
val_data['country '] = val_data['country']
del val_data['main_category']
del val_data['country']

# Remove severly extreme outliers 
val_data = val_data[val_data['Length'] < 3000.0]
#print(val_data['Length'].max())
#%%

# Create a list of the columns and remove all non-predictor columns
val_lst = list(val_data.columns)
val_lst.remove('launched')
val_lst.remove('deadline')
val_lst.remove('state')
val_lst.remove('dead')
val_lst.remove('launch')
val_lst.remove('Diff')
val_lst.remove('result')
val_lst.remove('pledged')
val_lst.remove('goal')
val_lst.remove('pledged ')
val_lst.remove('backers')

# Function to determine which of our predictors are categorical 
def cat_features(dataframe):
    td = pd.DataFrame({'a': [1,2,3], 'b': [1.0, 2.0, 3.0]})
    return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, 
                                          td['b'].dtype]), list(dataframe))

# Specify the x and the y variables and One-hot encode the categorical predictors 
val_data_x = val_data[val_lst]
val_data_x = pd.get_dummies(val_data_x, columns=list(cat_features(val_data_x[val_lst])))
val_data_y = val_data['result']

#print(len(list(val_data_x.columns)))
#for i in list(val_data_x.columns):
#    if i not in list(data_x.columns):
#        print(i)
#print(len(list(val_data_x.columns)))

#for i in list(data_x.columns):
#    if i not in list(val_data_x):
#        print(i)
#%%
val_data_x['country _N,"0'] = val_data_x['country _N,0"'] 
del val_data_x['country _N,0"']
del val_data_x['country _JP']
print(len(val_data_x.columns))

#%%
#for i in range(len(val_data_y)):
#    if val_data_y[i] == 'failure'
# Pass the Target Variable through the Label Encoding Pipeline 
val_data_y = le.fit_transform(val_data_y)

# Pass the training Predictors through the Preprocessing Pipeline 
val_x_pp = imp.transform(val_data_x)
val_x_pp = scaler.transform(val_x_pp)


#%%
val_y_hat = mod.predict(val_x_pp)

print("------Random Forest Model With n_estimators = 50, and max_depth = 8---------")
print("Accuracy Score: " + str(accuracy_score(val_y_hat, val_data_y)))
print("F1 Score: " + str(f1_score(val_y_hat, val_data_y)))

 #%%
val_k_fold_scores = cross_val_score(mod, val_x_pp, val_data_y, scoring='f1_macro', cv=k_fold)
print('CV Score (F1 Macro for K-Fold): ' + str(val_k_fold_scores))
print("Mean Score of The K-Fold CV: " + str(val_k_fold_scores.mean()))








