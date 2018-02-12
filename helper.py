import pandas as pd
import numpy as np
import math
import time
import os

from collections import Counter
from models import *





########################################## ADD NEW FEATURE ##########################################

def morning_afternoon(data_train, data_val, data_test):

	start_time =  time.time()
	list_morning = [u'hour_login_0', u'hour_login_1', u'hour_login_2', u'hour_login_3', u'hour_login_4', u'hour_login_5',
				u'hour_login_6', u'hour_login_7', u'hour_login_8', u'hour_login_9',u'hour_login_10', u'hour_login_11'] 
				
	list_afternoon = [u'hour_login_12', u'hour_login_13', u'hour_login_14', u'hour_login_15', u'hour_login_16', u'hour_login_17',
					  u'hour_login_18', u'hour_login_19', u'hour_login_20', u'hour_login_21', u'hour_login_22', u'hour_login_23']


	data_train['morning']   = data_train.apply(lambda x: np.sum([x[hour] for hour in list_morning]),   axis=1)
	data_train['afternoon'] = data_train.apply(lambda x: np.sum([x[hour] for hour in list_afternoon]), axis=1)

	data_val['morning']   = data_val.apply(lambda x: np.sum([x[hour] for hour in list_morning]),   axis=1)
	data_val['afternoon'] = data_val.apply(lambda x: np.sum([x[hour] for hour in list_afternoon]), axis=1)

	data_test['morning']   = data_test.apply(lambda x: np.sum([x[hour] for hour in list_morning]),   axis=1)
	data_test['afternoon'] = data_test.apply(lambda x: np.sum([x[hour] for hour in list_afternoon]), axis=1)

	data_train = data_train.drop(list_morning+list_afternoon, axis=1)
	data_val   = data_val.drop(  list_morning+list_afternoon, axis=1)
	data_test  = data_test.drop( list_morning+list_afternoon, axis=1)

	print 'Morning', round(time.time()-start_time)
	return data_train, data_val, data_test


	





########################################## STANDARDIZE THE COLUMNS + SPLIT BY USER ##########################################

# We want all the columns of the training dataset to be centered and standardized to have the relative importance of the features
# For every user, we want to have accesse to a list of all his sessions for faster computations


def helper(data_train, data_val, data_test, kind_features, session_OR_absence='session', name_dataset=''):

########################################## KEEP BASIC FEATURES ##########################################
	
	start_time = time.time()

	if kind_features=='basic':
		if name_dataset == 'last_fm':
			columns_to_keep = ['listener_id','genre', 'session_duration', 'log_session_duration', 'absence', 'past_session']

		else:
			columns_to_keep = [u'listener_id', u'session_duration', u'age', u'gender_FEMALE',
			   u'gender_MALE', u'state_CANCELLED', u'state_COMPLIMENTARY',
			   u'state_EXPIRED_SUBSCRIBER', u'state_REGISTERED', u'state_SUBSCRIBER',
			   u'state_SUSPENDED', u'state_VENDOR_BILLED_COMPLIMENTARY',
			   u'state_VENDOR_BILLED_SUBSCRIBER', u'category_SMARTPHONES',
			   u'category_WEB', u'network_type_CABLE_DSL', u'network_type_CELLULAR',
			   u'network_type_CORPORATE', u'network_type_DIALUP',
			   u'log_session_duration', u'last_session_time', u'absence_time']
		
		data_train = data_train[columns_to_keep]
		data_val   = data_val[columns_to_keep]
		data_test  = data_test[ columns_to_keep]


	if session_OR_absence=='session':
		y_data_users_val  = np.copy(data_val['session_duration'].values)
		y_data_users_test = np.copy(data_test['session_duration'].values)

	elif session_OR_absence=='absence':
		y_data_users_val  = np.copy(data_val['absence'].values)
		y_data_users_test = np.copy(data_test['absence'].values)

		#print 'Minimum absence: ' min(list(set(y_data_users_test) - set([0.0])))
		#print 'Minimum absence: ', min(y_data_users_test)


########################################## STANDARDIZE TRAIN ##########################################

	mean_train = []
	std_train  = []
	columns    = data_train.columns[1:] # not listener_id

	for col in columns: 
		data_train_col    = data_train[col].copy()
		mean_col, std_col = np.mean(data_train_col), np.std(data_train_col)+1e-10

		data_train[col] = data_train[col].apply(lambda x: (x-mean_col)/std_col)
		data_val[col]   = data_val[col].apply(  lambda x: (x-mean_col)/std_col)
		data_test[col]  = data_test[col].apply( lambda x: (x-mean_col)/std_col)

		mean_train.append(mean_col)
		std_train.append(std_col)
		print col, round(time.time()-start_time), np.mean(data_train[col])

	params_train_df = pd.DataFrame([mean_train, std_train], columns=columns, index=['mean', 'std']) 




########################################## SPLIT BY USER ##########################################

############# STORE INDEXES #############
	print session_OR_absence

	if session_OR_absence=='session':
		ids_train  = data_train['listener_id'].values
		ids_val    = data_val['listener_id'].values
		ids_test   = data_test['listener_id'].values

		y_data_users_train         = data_train['session_duration'].values
		y_log_data_users_train     = data_train['log_session_duration'].values



	elif session_OR_absence=='absence':
		ids_train  = data_train['listener_id'].values
		ids_val    = data_val['listener_id'].values
		ids_test   = data_test['listener_id'].values
		
		y_data_users_train         = data_train['absence'].values
		y_log_data_users_train     = data_train['log_absence'].values



	listeners_id_train = pd.unique(data_train['listener_id'])
	x_train_columns    = data_train.columns
	print round(time.time()-start_time)





	dict_train, dict_val, dict_test = {}, {}, {}

	for ids_set, dict_set in zip([ids_train, ids_val, ids_test], [dict_train, dict_val, dict_test]):
		for listener_id in listeners_id_train: dict_set[listener_id] = []

		for aux in range(ids_set.shape[0]):
			if aux%10000==0: print 'Sample ', aux, round(time.time()-start_time)
			dict_set[ids_set[aux]].append(aux)

	indexes_listeners_train  = [dict_train[key] for key in dict_train.keys()]
	number_points_user_train = [len(index) for index in indexes_listeners_train]
	indexes_listeners_val    = [dict_val[key]   for key in dict_val.keys()]
	indexes_listeners_test   = [dict_test[key]  for key in dict_test.keys()]


	number_points_user_val  = [len(index) for index in indexes_listeners_val]
	number_points_user_test = [len(index) for index in indexes_listeners_test]

	dict_val = Counter(number_points_user_val)
	dict_test = Counter(number_points_user_test)

	for i in range(20):
		print i, dict_val[i], dict_test[i]

	print stop
	
	return indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, data_train, y_data_users_train, y_log_data_users_train, number_points_user_train,  data_val, y_data_users_val, data_test, y_data_users_test






########################################## SAVE ALL DATA ##########################################

# We want to call the previous procedure only once -> then we will use the following load procedure

def process_and_save_data(kind_features, name_dataset, threshold_n_sessions, load_file, f, output_file='', load_best_tree=False):

	
	########################################## READ CSV ##########################################
	
	if True:
		#print name_dataset == 'last_fm_absence'
		if name_dataset in ['main_anonymized','enriched_sample_anonymized']:
			if threshold_n_sessions == 10:
				data_train = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_train.csv').drop(['Unnamed: 0'], axis=1)
				data_val   = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_val.csv').drop(['Unnamed: 0'], axis=1)
				data_test  = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_test.csv').drop(['Unnamed: 0'], axis=1)

			else:
				data_train = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_train.csv')
				data_val   = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_val.csv')
				data_test  = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'_test.csv')

			### Change hours to morning / afternoon ?
			data_train, data_val, data_test = morning_afternoon(data_train, data_val, data_test)

		elif name_dataset in ['last_fm', 'last_fm_absence']:
			data_train = pd.read_csv('../Dataset/'+name_dataset+'_train.csv')
			data_val   = pd.read_csv('../Dataset/'+name_dataset+'_val.csv')
			data_test  = pd.read_csv('../Dataset/'+name_dataset+'_test.csv')



		write_and_print('DATA SHAPE: '+str((data_train.shape, data_val.shape, data_test.shape)) , f)




		if kind_features in ['basic', 'advanced']: 

		########################################## PROCESS DATA ##########################################
			session_OR_absence = 'absence' if name_dataset=='last_fm_absence' else 'session'
			indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val,  x_data_users_test_STD, y_data_users_test = helper(data_train, data_val, data_test, kind_features, session_OR_absence=session_OR_absence, name_dataset=name_dataset)





		elif kind_features == 'xgboost_advanced':

		############ CREATE NEW FEATURES WITH XGBOOST ############

			indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, OLD_params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test = load_data('advanced', name_dataset, threshold_n_sessions, load_file)	
			data_train, data_val, data_test = create_xgboost_features(data_train, data_val, data_test, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, OLD_params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test , name_dataset, threshold_n_sessions, load_file, output_file, f, load_best_tree=load_best_tree)

			indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train,  x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test = helper(data_train, data_val, data_test, 'advanced')
			#for col in OLD_params_train_df.columns: params_train_df[col] = OLD_params_train_df[col]


	else:
		data_train = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_train'+'.csv')
		data_val   = pd.read_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_val'+'.csv')
		#data_test  = pd.read_csv( '../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_test'+'.csv')
		
		data_val  = data_val.iloc[:,:data_train.shape[1]]
		data_test = data_test.iloc[::data_train.shape[1]]
		print data_train.shape, data_val.shape, data_test.shape
		print data_train.columns
		print data_val.columns

		indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, data_train, y_data_users_train, y_log_data_users_train, number_points_user_train,  data_val, y_data_users_val, data_test, y_data_users_test = helper(data_train, data_val, data_test, 'advanced')
			




########################################## SAVE DATA ##########################################
	
	name_folder = load_file+'/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'/'+kind_features
	if not os.path.exists(name_folder): os.makedirs(name_folder)

	np.save(name_folder+'/indexes_listeners_train', indexes_listeners_train)
	np.save(name_folder+'/indexes_listeners_val'  , indexes_listeners_val)
	np.save(name_folder+'/indexes_listeners_test' , indexes_listeners_test)

	params_train_df.to_csv(name_folder+'/params_train_df.csv')

	np.save(name_folder+'/x_train_columns', x_train_columns)
	np.save(name_folder+'/y_data_users_train', y_data_users_train)
	np.save(name_folder+'/y_log_data_users_train', y_log_data_users_train)
	np.save(name_folder+'/number_points_user_train', number_points_user_train)
	np.save(name_folder+'/y_data_users_val', y_data_users_val)
	np.save(name_folder+'/y_data_users_test', y_data_users_test)
	

	np.save(name_folder+'/x_data_users_train', data_train.drop(['listener_id', 'session_duration', 'log_session_duration'], axis=1).values)
	np.save(name_folder+'/x_data_users_val_STD', data_val.drop(['listener_id',   'session_duration', 'log_session_duration'], axis=1).values)
	np.save(name_folder+'/x_data_users_test_STD', data_test.drop(['listener_id',  'session_duration', 'log_session_duration'], axis=1).values)







########################################## LOAD ALL DATA ##########################################

def load_data(kind_features, name_dataset, threshold_n_sessions, load_file):

	name_folder = load_file+'/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'/'+kind_features

	indexes_listeners_train = list( np.load(name_folder+'/indexes_listeners_train.npy') )
	indexes_listeners_val   = list( np.load(name_folder+'/indexes_listeners_val.npy') )
	indexes_listeners_test  = list( np.load(name_folder+'/indexes_listeners_test.npy' ) )

	indexes_listeners_train = [[int(a) for a in aa] for aa in indexes_listeners_train]
	indexes_listeners_val   = [[int(a) for a in aa] for aa in indexes_listeners_val]
	indexes_listeners_test  = [[int(a) for a in aa] for aa in indexes_listeners_test]

	params_train_df = pd.read_csv(name_folder+'/params_train_df.csv').drop(['Unnamed: 0'], axis=1)
	params_train_df.index = ['mean','std']	

	x_train_columns    = list( np.load(name_folder+'/x_train_columns.npy') )
	x_data_users_train = np.load(name_folder+'/x_data_users_train.npy') 
	y_data_users_train = np.load(name_folder+'/y_data_users_train.npy') 
	y_log_data_users_train   = np.load(name_folder+'/y_log_data_users_train.npy') 
	number_points_user_train = np.load(name_folder+'/number_points_user_train.npy') 

	x_data_users_val_STD  = np.load(name_folder+'/x_data_users_val_STD.npy') 
	y_data_users_val      = np.load(name_folder+'/y_data_users_val.npy') 
	
	x_data_users_test_STD = np.load(name_folder+'/x_data_users_test_STD.npy') 
	y_data_users_test     = np.load(name_folder+'/y_data_users_test.npy') 

	return indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test







def create_xgboost_features(data_train, data_val, data_test, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val, y_data_users_val, x_data_users_test, y_data_users_test , name_dataset, threshold_n_sessions, load_file, output_file, f, load_best_tree=False):

########################################## RUN XGBOOST WITH TREES SIZE 2 ##########################################	

	if not load_best_tree:
		list_llambda = [2,4,6,8,10]
		baseline_val_MAE, baseline_test_MAE, _            = baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test)
		all_ratios_val, all_ratios_test, dict_best_params = loop_alternative_minimization_xgboost(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, f, baseline_val=baseline_val_MAE, baseline_test=baseline_test_MAE, only_size_2=True)

		np.save(output_file+'/../xgboost_advanced/EB-xgboost/all_ratios_xgboost_advanced_features_trees_size2_VAL',  all_ratios_val)
		np.save(output_file+'/../xgboost_advanced/EB-xgboost/all_ratios_xgboost_advanced_features_trees_size2_TEST', all_ratios_test)

		dict_best_params['model'].dump_model(output_file+'/../xgboost_advanced/EB-xgboost/correct_model_alternative_xgboost_trees_size2.txt')
		best_model_str = dict_best_params['model'].get_dump()

		write_and_print('\nBest VAL  MAE ALT_MIN XGBOOST TREES SIZE 2: '+str(round(np.min(all_ratios_val), 3)), f)
		write_and_print('Best TEST MAE ALT_MIN XGBOOST TREES SIZE 2: ' +str(round(np.min(all_ratios_test), 3)), f)

		write_and_print('\nParams ALT_MIN XGBOOST TREES SIZE 2: '+str(dict_best_params['params']), f)
		write_and_print('Correct TEST MAE ALT_MIN XGBOOST TREES SIZE 2: '+str(round(dict_best_params['error'], 3)), f)

	else:
		model = open(output_file+'/../EB-xgboost/correct_model_alternative_xgboost_trees_size2.txt')
		write_and_print('Model loaded', f)

		best_model_str = []
		new_tree =''

		for line in model:
			if line[:7]=='booster':
				if len(new_tree)>0: best_model_str.append(new_tree)
				new_tree =''
			elif line!='\n':
				new_tree += line+'\r'
				



########################################## CREATE NEW FEATURES WITH XGBOOST ##########################################

	all_list_trees = []

	for tree in best_model_str:
		aux = -1
		list_tree = []
		for node in tree.split('\n'):
			try: #one missing leaf
				aux += 1 
				if aux in [0, 1, 4]: #trees of size 2
					leaf = node.split('[')[1].split(']')[0]
					leaf = leaf.split('<')

					list_tree.append( (x_train_columns[ int(leaf[0][1:]) ],  float(leaf[1])) )
			except:
				continue
		
		if list_tree not in all_list_trees and len(list_tree)==3:
			all_list_trees.append(list_tree)  



	for i in range(len(all_list_trees)):
		tree = all_list_trees[i]
		print 'Tree number ',i
	
		for data in [data_train, data_val, data_test]:
			try:
				data['tree_'+str(i)+'_feature_0'] =  data.apply(lambda x: int(x[tree[0][0]] < tree[0][1]  and  x[tree[1][0]] <= tree[1][1]),   axis=1)
				data['tree_'+str(i)+'_feature_1'] =  data.apply(lambda x: int(x[tree[0][0]] < tree[0][1]  and  x[tree[1][0]]  > tree[1][1]),   axis=1)
				data['tree_'+str(i)+'_feature_2'] =  data.apply(lambda x: int(x[tree[0][0]] > tree[0][1]  and  x[tree[2][0]] <= tree[2][1]),   axis=1)
				data['tree_'+str(i)+'_feature_3'] =  data.apply(lambda x: int(x[tree[0][0]] > tree[0][1]  and  x[tree[2][0]]  > tree[2][1]),   axis=1)
			except:
				continue

	data_train.to_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_train'+'.csv', index=False)
	data_val.to_csv('../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_val'+'.csv',     index=False)
	data_test.to_csv( '../Dataset/'+name_dataset+'_processed_'+str(threshold_n_sessions)+'xgboost_features_test'+'.csv',  index=False)
	return data_train, data_val, data_test
								   












