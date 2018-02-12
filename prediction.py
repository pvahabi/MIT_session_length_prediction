import pandas as pd
import numpy as np
import math
import os

from run_models import *
from plot import *
from helper import *
import matplotlib.pyplot as plt

from collections import Counter




# KIND_FEATURES in {'basic', 'advanced', 'xgboost_advanced'}:
	# - basic:            original features
	# - advanced:         additional ones
	# - xgboost_advanced: xgboost run on advanced

#NAME_DATASET in {'enriched_sample_anonymized', 'main_anonymized'}


THRESHOLD_N_SESSIONS = 1








###### EMPIRICAL BAYES WITH COVARIATES - RIDGE / LASSO ######


def prediction_sequential(kind_features, name_dataset, load_file, output_file, do_processing, run_models, load_best_tree):

########################################## READ DATA #################x_data_users_tes#########################

###### Results
	f = open(output_file+'/results_sequential.txt', 'w')

###### DO WE HAVE TO PROCESS THE DATA
	if do_processing: process_and_save_data(kind_features, name_dataset, THRESHOLD_N_SESSIONS, load_file, f, output_file=output_file, load_best_tree=load_best_tree)

###### Load
	if run_models:
		indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test = load_data(kind_features, name_dataset, THRESHOLD_N_SESSIONS, load_file)

		#fig = plt.figure(figsize=(15,5))
		#n, bins, patches = plt.hist(y_log_data_users_train, 100, range=(-4,4))
		#plt.savefig('../last_fm_log_session_length.pdf')

		#print len(x_data_users_train), len(x_data_users_val_STD), len(x_data_users_test_STD), len(indexes_listeners_train), len(x_train_columns)


	########################################## TEST ALL SIMPLE MODELS ##########################################

	###### BASELINE ######
		baseline_val_MAE, baseline_test_MAE, dict_baseline_pairs_sessions_errors = run_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test, f, output_file=output_file)

		#baseline_val_MAE, baseline_test_MAE =  len(x_data_users_test_STD), len(x_data_users_test_STD)
		


	###### LOG BASELINE ######
		run_log_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, f)

		

	###### XGBOOST ######

	### First model predict the session
		run_xgboost(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, y_log_data_users_train, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, params_train_df, baseline_val_MAE, baseline_test_MAE, output_file, f, predict_log=True)

	### Second model predicts the logs
		#run_xgboost(data_train, data_test, data_train_std, data_test_std, params_train_df, baseline_test_MAE, output_file, f, predict_log=True)



	###### EMPIRICAL BAYES ######

	### With range of alpha
		run_EB_log(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, number_points_user_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, theoretical_alpha=False)

	### With theoretical alpha
		run_EB_log(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, number_points_user_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, theoretical_alpha=True)



	###### REGRESSION ######
		run_ridge(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, params_train_df, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, theoretical_alpha=False)




		f.close()
	










###### EMPIRICAL BAYES WITH COVARIATES - RIDGE / LASSO ######

def prediction_EB_penalization(kind_features, name_dataset, load_file, output_file, list_llambda):

	if not os.path.exists(output_file): os.makedirs(output_file)

	f = open(output_file+'/results_EB_penalization.txt', 'w')
	indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test = load_data(kind_features, name_dataset, THRESHOLD_N_SESSIONS, load_file)

	#print x_train_columns.shape, x_data_users_val_STD.shape, x_data_users_test_STD.shape
	baseline_val_MAE, baseline_test_MAE, dict_baseline_pairs_sessions_errors = run_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test, f)

	run_EB_penalization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, penalization='l1')
	run_EB_penalization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, penalization='l2')
	
	if kind_features=='xgboost_advanced': 
		P = len(x_train_columns)
		range_K = [20,25,30,35]+[P]
		range_K = range_K[::-1]
		run_EB_penalization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, penalization='l2', use_L0=True, range_K=range_K)
	f.close()












###### EMPIRICAL BAYES WITH COVARIATES - XGBOOST ######

def prediction_EB_xgboost(kind_features, name_dataset, load_file, output_file, list_llambda):

	if not os.path.exists(output_file): os.makedirs(output_file)

	f = open(output_file+'/results_EB_xgboost.txt', 'w')
	indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, params_train_df, x_train_columns, x_data_users_train, y_data_users_train, y_log_data_users_train, number_points_user_train, x_data_users_val_STD, y_data_users_val, x_data_users_test_STD, y_data_users_test = load_data(kind_features, name_dataset, THRESHOLD_N_SESSIONS, load_file)

	baseline_val_MAE, baseline_test_MAE, dict_baseline_pairs_sessions_errors = run_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test, f)

	run_EB_xgboost(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f)
	f.close()

	






































