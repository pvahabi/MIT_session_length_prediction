import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV 
from sklearn.model_selection import ParameterGrid


name_file = 'main_anonymized'







#### READ DATA ####

columns = ['listener_device_session_id', 'listener_id', 'session_duration',  'start_time', 'end_time',  'age', 'gender', 'state', 'category',  'user_seed', 'network_type', 'skips',   'station_changes', 'thumbs_down', 'thumbs_up', 
		   'num_ads', 'track_hours', 'ad_hours']

#Drop nan values
data = pd.read_table('../Dataset/'+name_file+'txt', names=columns).dropna()
data = data.sort_values(['start_time'], ascending=True)
data.index = range(data.shape[0])
print 'Original shape %s'%(str(data.shape))

data = data[['listener_id', 'session_duration', 'start_time', 'end_time', 'age', 'gender', 'state', 'category', 'network_type']]
data = pd.get_dummies(data, columns=['gender', 'state', 'category', 'network_type'])
data = data.sample(frac=1).reset_index(drop=True)





### LIMIT FOR USER BASED MODEL + ADD TIMING FEATURES ####

threshold_n_users = 20

listeners_id = pd.unique(data['listener_id'])
new_data     = pd.DataFrame() 

for idx in listeners_id:
	data_user = data[data['listener_id'] == idx]
	data_user = data_user.sort_values(['start_time'], ascending=True)

	if data_user.shape[0] > threshold_n_users:
		data_user['end_time_shift']    = data_user['end_time'].shift(1)
		data_user['start_time_shift']  = data_user['start_time'].shift(1)
		data_user['last_session_time'] = data_user.apply(lambda x: (pd.Timestamp(x['end_time_shift']) - pd.Timestamp(x['start_time_shift'])).total_seconds(), axis=1)
		data_user['absence_time']      = data_user.apply(lambda x: (pd.Timestamp(x['start_time'])     - pd.Timestamp(x['end_time_shift'])).total_seconds() ,  axis=1)
		arr    = np.arange(data_user.shape[0])
		arr[0] = 1
		data_user['averaged_past_sessions'] = pd.Series(np.divide(np.cumsum(data_user['last_session_time'].shift(1)), arr))
		data_user = data_user.fillna(0)
		new_data  = pd.concat([new_data, data_user], axis=0)


data      = new_data.sort_values(['start_time'], ascending=True)
data.index = range(data.shape[0])
print 'Train shape %s'%(str(data.shape))









### CREATE TRAIN AND TEST ####

N = data.shape[0]
data_train = data[:int(.8*N)]
data_test  = data[int(.8*N):]

listeners_id_train = pd.unique(data_train['listener_id'])
listeners_id_test  = pd.unique(data_test[ 'listener_id'])

#Make sur all users from test belong to one of the training
for idx in listeners_id_test:
    if idx not in listeners_id_train:
        data_user_test = data_test[data_test['listener_id'] == idx]
        data_test      = data_test.drop(data_user_test.index)
        

data_test       = data_test.sort_values(['start_time'], ascending=True)
data_test.index = range(data_test.shape[0])







### FIRST MODEL: BASELINE ####


listeners_id         = pd.unique(data_train['listener_id'])
mean_listeners_train = []

for idx in listeners_id:
    mean_listener = np.mean(data_train[data_train['listener_id'] == idx]['session_duration'])
    mean_listeners_train.append(mean_listener)
    
baseline_train           = pd.DataFrame([listeners_id, mean_listeners_train], index=['listener_id_train','average_length_train']).T
baseline_test_preditions = [float(baseline_train[baseline_train['listener_id_train'] == idx]['average_length_train']) for idx in data_test['listener_id'].values]
baseline_test_MAE_1      = np.sum(np.abs(baseline_test_preditions - data_test['session_duration']))  
baseline_test_MAE_2      = np.sum(np.abs(data_test['averaged_past_sessions'] - data_test['session_duration']))  

print 'Baseline comparison %s'%(str( baseline_test_MAE_2 / baseline_test_MAE_1))







### SECOND MODEL: USER BASED LIN REG ####

user_based_test_MAE = 0

for idx in listeners_id:
    data_train_reduced = data_train[data_train['listener_id'] == idx]
    data_test_reduced  = data_test[ data_test[ 'listener_id'] == idx]
    
    X_train, y_train   = data_train_reduced.drop(['session_duration', 'listener_id'], axis=1), data_train_reduced['session_duration'] 
    X_test,  y_test    = data_test_reduced.drop([ 'session_duration', 'listener_id'], axis=1), data_test_reduced[ 'session_duration'] 
    
    if len(y_test) > 0:
        estimator = LinearRegression()
        estimator.fit(X_train, y_train)
        user_based_test_predictions = estimator.predict(X_test)  
        user_based_test_MAE        += np.sum(np.abs(user_based_test_predictions - y_test))

print 'User based MAE with baseline 2: %s'%(user_based_test_MAE/baseline_test_MAE_2)








### THIRD MODEL: TRAIN AND TUNE ON WHOLE DATASET ####

X_train, y_train = data_train.drop(['session_duration', 'listener_id'], axis=1), data_train['session_duration']
X_test , y_test = data_test.drop([  'session_duration', 'listener_id'], axis=1), data_test['session_duration']


test_params = {
    'learning_rate': [10e-2, 10e-3],
    'max_depth': [4,6,8],
    'n_estimators': [10, 50, 100]
}

params_grid = ParameterGrid(test_params)
all_MAE     = [] 

for params in params_grid:
    a,b,c = params.values()
    
    xgb_model = XGBRegressor(n_estimators=a, learning_rate=b, max_depth=c)
    xgb_model.fit(X_train, y_train)
    xgb_all_user_predictions = xgb_model.predict(X_test)
    xgb_all_user_MAE         = np.sum(np.abs(xgb_all_user_predictions - y_test))
    
    all_MAE.append(xgb_all_user_MAE/baseline_test_MAE_2)

argmin = np.argmin(all_MAE)    
print 'Params: %s'%(params_grid[argmin])
print 'User based MAE with baseline 2: %s\n'%(all_MAE[argmin])







### FOURTH MODEL: CLUSTERING ON USERS BASED MODELS ###

# Cluster on data train

best_k = 20
data_kmeans = baseline_train[columns_kmeans].values.reshape(baseline_train.shape[0], len(columns_kmeans))
kmeans      = KMeans(n_clusters = best_k).fit(data_kmeans)
baseline_train['Kmeans'] = kmeans.predict(data_kmeans)



# Tune parameters for each cluster

xgb_cluster_user_MAE = 0

for cluster in range(best_k):
    
    data_train_Kmeans = data_train[data_train['Kmeans']==cluster].drop(['Kmeans'], axis=1)
    data_test_Kmeans  = data_test[ data_test[ 'Kmeans']==cluster].drop(['Kmeans'], axis=1)
    print data_train_Kmeans.shape, data_test_Kmeans.shape

    X_train, y_train = data_train_Kmeans.drop(['session_duration', 'listener_id'], axis=1), data_train_Kmeans['session_duration']
    X_test , y_test  = data_test_Kmeans.drop([ 'session_duration', 'listener_id'], axis=1), data_test_Kmeans[ 'session_duration']

    test_params = {
        'learning_rate': [10e-2, 10e-3],
        'max_depth': [4,6,8],
        'n_estimators': [10, 50, 100]
    }
    params_grid = ParameterGrid(test_params)
    all_MAE     = []
    for params in params_grid:
        a,b,c = params.values()
        xgb_model = XGBRegressor(n_estimators=a, learning_rate=b, max_depth=c)
        xgb_model.fit(X_train, y_train)

        xgb_cluster_user_predictions = xgb_model.predict(X_test)
        xgb_cluster_user_MAE         = np.sum(np.abs(xgb_cluster_user_predictions - y_test))
        all_MAE.append(xgb_cluster_user_MAE)
    
    argmin = np.argmin(all_MAE)    
    print 'Params: %s'%(params_grid[argmin])
    xgb_cluster_user_MAE += all_MAE[argmin]
    
print 'User based MAE with baseline 2: %s\n'%(xgb_cluster_user_MAE/baseline_test_MAE_2)






