import pandas as pd
import numpy as np
import math

THRESHOLD_N_SESSIONS = 1


def process_dataset(name_dataset):

########################################## READ DATA ##########################################

	columns = ['listener_device_session_id', 'listener_id', 'session_duration',  'start_time', 'end_time',  'age', 'gender', 'state', 'category',  
				'user_seed', 'network_type', 'skips',   'station_changes', 'thumbs_down', 'thumbs_up', 'num_ads', 'track_hours', 'ad_hours']

### Drop nan values
	data = pd.read_table('../Dataset/'+name_dataset+'.txt', names=columns).dropna()


### Consider features known at the beginning of the session
	data = data[['listener_id', 'session_duration', 'start_time', 'end_time', 'age', 'gender', 'state', 'category', 'network_type']]


### Sort by dates
	data = data.sort_values(['start_time'], ascending=True)
	data.index = range(data.shape[0])
	print 'Dataset shape:', data.shape


### Convert binary variables
	data = pd.get_dummies(data, columns=['gender', 'state', 'category', 'network_type'])
	data = data.sample(frac=1).reset_index(drop=True)





########################################## CREATE FEATURES ##########################################

### Add log of sessions
	data = data[data['session_duration']>0]
	data['log_session_duration'] = data.apply(lambda x: math.log(x['session_duration']), axis=1)


	# We delete users with a small number of observations
	# For the remaining ones, we create the following features for each session:
		# - the last_session_time
		# - the absence time
		# - the averaged time of the past sessions



	listeners_id = pd.unique(data['listener_id'])
	new_data     = pd.DataFrame() 

	aux = -1
	for idx in listeners_id:
		aux += 1
		print aux

	### Sort by user and time
		data_user = data[data['listener_id'] == idx]
		data_user = data_user.sort_values(['start_time'], ascending=True)

		if data_user.shape[0] > THRESHOLD_N_SESSIONS:
			data_user['hour_login']        = data_user.apply(lambda x: pd.Timestamp(x['start_time']).hour, axis=1)

			data_user['end_time_shift']    = data_user['end_time'].shift(1)
			data_user['start_time_shift']  = data_user['start_time'].shift(1)
			
		### Last session time
			data_user['last_session_time']     = data_user.apply(lambda x: (pd.Timestamp(x['end_time_shift']) - pd.Timestamp(x['start_time_shift'])).total_seconds(), axis=1)
			data_user = data_user.fillna(1)
			data_user['last_session_time']     = data_user['last_session_time'].apply(lambda x: max(1,x))
			data_user['log_last_session_time'] = data_user.apply(lambda x: math.log(float(x['last_session_time'])) , axis=1)
			

		### Absence time
			data_user['absence_time']     = data_user.apply(lambda x: (pd.Timestamp(x['start_time'])     - pd.Timestamp(x['end_time_shift'])).total_seconds() ,  axis=1)
			data_user = data_user.fillna(1)
			data_user['absence_time']     = data_user['absence_time'].apply(lambda x: max(1,x))
			data_user['log_absence_time'] = data_user.apply(lambda x: math.log(float( x['absence_time'])), axis=1)
			


		### Averaged time of past sessions
			arr    = np.arange(data_user.shape[0])
			arr[0] = 1
			data_user['averaged_past_sessions']     = pd.Series(np.divide(np.cumsum(data_user['last_session_time']), arr))
			data_user['log_averaged_past_sessions'] = pd.Series(np.divide(np.cumsum(data_user['log_last_session_time']), arr))
			data_user = data_user.fillna(1)
			
		### Concatenate
			new_data  = pd.concat([new_data, data_user], axis=0)

			if aux % 5000 == 0:
				new_data.to_csv('../Dataset/'+name_dataset+'_processed_'+str(THRESHOLD_N_SESSIONS)+'_'+str(aux)+'.csv')


### Sort by time
	new_data  = pd.get_dummies(new_data, columns=['hour_login'])
	data      = new_data.sort_values(['start_time'], ascending=True)
	data.index = range(data.shape[0])
	print 'Dataset processed shape:', data.shape 


### Save
	data.to_csv('../Dataset/'+name_dataset+'_processed_'+str(THRESHOLD_N_SESSIONS)+'.csv')
	return data







def split_train_val_test(data, name_dataset, add_features=False):

### Split for a given ratio
	RATIO_TRAIN = .8
	RATIO_VAL   = .1

	if add_features:
		data['region_averaged_past_sessions'] = data['averaged_past_sessions'].apply(f)
		data = pd.get_dummies(data, columns=['region_averaged_past_sessions'])

	N = data.shape[0]

	data_train      = data[:int(RATIO_TRAIN*N)]
	data_val        = data[int(RATIO_TRAIN*N):int( (RATIO_TRAIN+RATIO_VAL)*N) ]
	data_test       = data[int((RATIO_TRAIN+RATIO_VAL)*N): ]
	data_val.index  = range(data_val.shape[0])
	data_test.index = range(data_test.shape[0])
	
	print 'Train shape: ', data_train.shape
	print 'Val shape: ',   data_val.shape
	print 'Test shape : ', data_test.shape

	print data_train.columns


### Only keep users in test which appear in train
	listeners_id_train = pd.unique(data_train['listener_id'])
	listeners_id_val   = pd.unique(data_val[  'listener_id'])
	listeners_id_test  = pd.unique(data_test[ 'listener_id'])

	for idx in list(listeners_id_val) + list(listeners_id_test):
	    if idx not in listeners_id_train:
	        data_user_test = data_test[data_test['listener_id'] == idx]
	        data_test      = data_test.drop(data_user_test.index)
	        
	        data_user_val  = data_val[data_val['listener_id'] == idx]
	        data_val       = data_val.drop(data_user_val.index)


### Drop times
	data_train = data_train.drop(['start_time', 'end_time', 'start_time_shift', 'end_time_shift'], axis=1)
	data_val   = data_val.drop([  'start_time', 'end_time', 'start_time_shift', 'end_time_shift'], axis=1)
	data_test  = data_test.drop([ 'start_time', 'end_time', 'start_time_shift', 'end_time_shift'], axis=1)


### Save
	data_train.to_csv('../Dataset/'+name_dataset+'_processed_'+str(THRESHOLD_N_SESSIONS)+'_train'+'.csv', index=False)
	data_val.to_csv(  '../Dataset/'+name_dataset+'_processed_'+str(THRESHOLD_N_SESSIONS)+'_val'+'.csv',   index=False)
	data_test.to_csv( '../Dataset/'+name_dataset+'_processed_'+str(THRESHOLD_N_SESSIONS)+'_test'+'.csv',  index=False)






if False:
	#name_dataset = 'enriched_sample_anonymized'
	name_dataset = 'main_anonymized'
	data  = process_dataset(name_dataset)
	split_train_test(data, name_dataset)



if True:
	name_dataset = 'main_anonymized'
	data  = pd.read_csv('../Dataset/'+name_dataset+'_processed_1.csv').drop(['Unnamed: 0'], axis=1)
	split_train_val_test(data, name_dataset, add_features=False)








