import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import ParameterGrid

from DFO_L0_LQ import *


N_ALPHA = 40


########################################## AVERAGED ESTIMATORS ##########################################

### Baseline estimator: mean of the train set 
### Don't forget: the train data has been standardized

def baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test):

	all_means_users_train = [np.mean( y_data_users_train[ indexes_listeners_train[i] ] ) for i in range(len(indexes_listeners_train))]
	dict_pairs_sessions_errors = {}

	baseline_val_MAE  = []
	baseline_test_MAE = []

	for i in range(len(indexes_listeners_test)):  
		baseline_train     = params_train_df['session_duration']['mean'] + params_train_df['session_duration']['std']*all_means_users_train[i] 
		
		y_data_user_val    = y_data_users_val[ indexes_listeners_val[i] ]
		sum_error_user     = [abs(baseline_train - y_data) for y_data in y_data_user_val]
		baseline_val_MAE.extend( sum_error_user )

		y_data_user_test   = y_data_users_test[ indexes_listeners_test[i] ]
		sum_error_user     = [abs(baseline_train - y_data) for y_data in y_data_user_test]
		baseline_test_MAE.extend( sum_error_user )
		
		n_user = len(indexes_listeners_train[i])
		dict_pairs_sessions_errors[n_user]  = sum_error_user if n_user not in dict_pairs_sessions_errors.keys() else dict_pairs_sessions_errors[n_user]+sum_error_user


	baseline_val_MAE  = np.median(baseline_val_MAE)
	baseline_test_MAE = np.median(baseline_test_MAE)
	return baseline_val_MAE, baseline_test_MAE, dict_pairs_sessions_errors




### Mean of the log of the the train set 
### Take the exponential

def baseline_log(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors):

	all_means_users_train      = [np.mean(y_log_data_users_train[ indexes_listeners_train[i] ]) for i in range(len(indexes_listeners_train))]
	dict_pairs_sessions_errors = {}

	log_baseline_val_MAE  = []
	log_baseline_test_MAE = []

	for i in range(len(indexes_listeners_test)):  
		log_baseline_train      = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*all_means_users_train[i] 
		
		y_data_user_val         = y_data_users_val[ indexes_listeners_val[i] ]
		sum_error_user          = [abs(math.exp(log_baseline_train) - y_data) for y_data in y_data_user_val]
		log_baseline_val_MAE.extend( sum_error_user )

		y_data_user_test        = y_data_users_test[ indexes_listeners_test[i] ]
		sum_error_user          = [abs(math.exp(log_baseline_train) - y_data) for y_data in y_data_user_test]
		log_baseline_test_MAE.extend( sum_error_user )

		n_user = len(indexes_listeners_train[i])
		
		dict_pairs_sessions_errors[n_user] =  sum_error_user if n_user not in dict_pairs_sessions_errors.keys() else dict_pairs_sessions_errors[n_user]+sum_error_user
		#dict_pairs_sessions_errors[n_user] =  sum_error_user/(1e-10 + dict_baseline_pairs_sessions_errors[n_user]) if n_user not in dict_pairs_sessions_errors.keys() else dict_pairs_sessions_errors[n_user]+sum_error_user/(1e-10 + dict_baseline_pairs_sessions_errors[n_user])
	
	log_baseline_val_MAE  = np.median(log_baseline_val_MAE)
	log_baseline_test_MAE = np.median(log_baseline_test_MAE)
	return log_baseline_val_MAE, log_baseline_test_MAE, dict_pairs_sessions_errors







########################################## RIDGE ESTIMATORS ##########################################


def ridge(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, params_train_df, x_data_users_val, y_data_users_val, x_data_users_test, y_data_users_test, dict_baseline_pairs_sessions_errors, f, baseline_val=1, baseline_test=1, theoretical_alpha=False, is_test_standardized=False):

	### SVD for Ridge
	U, Diag, _  = np.linalg.svd(np.dot(x_data_users_train.T, x_data_users_train))


	### Which alpha ? -> range of values or theoretical ?
	if not theoretical_alpha:
		alpha_ridge_max  = power_method(x_data_users_train) 
		list_alpha_ridge = [alpha_ridge_max*0.7**i for i in range(N_ALPHA)]

	else:
		y_norm  = np.linalg.norm(y_log_data_users_train)**2
		y_sum   = np.sum(y_log_data_users_train)**2
		n_user  = len(y_log_data_users_train)

		X_norm  = np.sum(np.linalg.norm(x_data_users_train, axis=0)**2)
		#X_sum   = np.sum( np.dot(x_data_users_train.T, x_data_users_train) )
		X_sum   = 0

		est_sigma2_2 = (y_sum  - y_norm) / float( X_sum-X_norm )
		est_sigma1_2 = (y_norm - X_norm*est_sigma2_2)/float(n_user) 

		print y_norm, X_norm, est_sigma2_2, y_sum
		
		list_alpha_ridge = [est_sigma1_2/est_sigma2_2]

	#print list_alpha_ridge


	### Main loop
	all_ratios_val  = []
	all_ratios_test = []

	old_beta_train_std            = np.zeros(np.array(x_data_users_train).shape[1])
	#all_means_x_data_users_train  = np.array([ params_train_df[col]['mean'] for col in x_train_columns])
	ones_N = np.ones(x_data_users_train.shape[0])


	### Results
	best_ratio_val                  = 1e6
	best_ratio_test                 = 1e6
	best_beta_std, best_alpha       = [], (0,0)
	best_dict_pairs_sessions_errors = {}


	aux = 0
	for alpha_ridge in list_alpha_ridge:
		aux += 1
		#estimator  = Ridge(alpha=alpha_ridge, fit_intercept=False, solver='svd')
		#estimator.fit(x_data_users_train, y_log_data_users_train)
		#beta_train_std = np.copy(estimator.coef_)

		beta_train_std = np.dot(np.dot(np.dot(U, np.diag(1/(Diag + alpha_ridge))), U.T), np.dot(x_data_users_train.T, y_log_data_users_train))


	### Compute the test accuracy ->is the data standardized ??
		if not is_test_standardized:
			beta_train = beta_train_std * np.array([(params_train_df['log_session_duration']['std']/params_train_df[col]['std']) for col in x_train_columns])

	### Test results
		ratio_cov_EB_val_MAE       = []
		ratio_cov_EB_test_MAE      = []
		dict_pairs_sessions_errors = {}

		for i in range(len(indexes_listeners_test)):  

		

		####### VAL ########

		### Data user
			x_data_user_val  = x_data_users_val[indexes_listeners_val[i]]
			y_data_user_val  = y_data_users_val[indexes_listeners_val[i]]
			n_user_val       = len(x_data_user_val)

			if n_user_val>0:
				if not is_test_standardized:
					add = params_train_df['log_session_duration']['mean'] - np.dot(all_means_x_data_users_train, beta_train) 
					sum_error_user = np.abs( np.exp(np.dot(x_data_user_val, beta_train) + add*ones_N[:n_user_val] ) - y_data_user_val )
				else:
					pred_EB_val    = params_train_df['log_session_duration']['std'] * np.dot(x_data_user_val, beta_train_std) + params_train_df['log_session_duration']['mean'] 
					sum_error_user = np.abs( np.exp(pred_EB_val) - y_data_user_val )
			else:
				sum_error_user = []

			ratio_cov_EB_val_MAE.extend( sum_error_user )



		####### TEST ########

		### Data user
			x_data_user_test = x_data_users_test[indexes_listeners_test[i]]
			y_data_user_test = y_data_users_test[indexes_listeners_test[i]]
			n_user_test      = len(x_data_user_test)

			if n_user_test>0:
				if not is_test_standardized:
					add = params_train_df['log_session_duration']['mean'] - np.dot(all_means_x_data_users_train, beta_train) 
					sum_error_user = np.abs( np.exp(np.dot(x_data_user_test, beta_train) + add*ones_N[:n_user_test] ) - y_data_user_test )
				else:
					pred_EB_test   = params_train_df['log_session_duration']['std'] * np.dot(x_data_user_test, beta_train_std) + params_train_df['log_session_duration']['mean'] 
					sum_error_user = np.abs( np.exp(pred_EB_test) - y_data_user_test )
			else:
				sum_error_user = []

			
			n_user_train = len(y_log_data_users_train[indexes_listeners_train[i]])

			if n_user_train not in dict_pairs_sessions_errors.keys():
				dict_pairs_sessions_errors[n_user_train] = list(sum_error_user)
			else:
				dict_pairs_sessions_errors[n_user_train].extend( sum_error_user ) 
		

			#dict_pairs_sessions_errors[n_user_train] =  sum_error_user/(dict_baseline_pairs_sessions_errors[n_user_train] + 1e-10) if n_user_train not in dict_pairs_sessions_errors.keys() else dict_pairs_sessions_errors[n_user_train]+sum_error_user/(dict_baseline_pairs_sessions_errors[n_user_train] + 1e-10)
			ratio_cov_EB_test_MAE.extend( sum_error_user )

		ratio_cov_EB_val_MAE   = np.median(ratio_cov_EB_val_MAE)
		ratio_cov_EB_test_MAE  = np.median(ratio_cov_EB_test_MAE)


		ratio_cov_EB_val_MAE /= baseline_val
		ratio_cov_EB_test_MAE /= baseline_test
		#write_and_print('Test prediction: '+str(ratio_cov_EB_test_MAE) , f)

##################################### CAREFULLL ############################
		if ratio_cov_EB_val_MAE < best_ratio_val:
			best_ratio_val  = ratio_cov_EB_val_MAE

			best_ratio_test = ratio_cov_EB_test_MAE
			best_beta_std   = np.copy(beta_train_std)
			best_alpha      = alpha_ridge
			best_dict_pairs_sessions_errors = dict_pairs_sessions_errors

		all_ratios_val.append(ratio_cov_EB_val_MAE)
		all_ratios_test.append(ratio_cov_EB_test_MAE)

	dict_best = {'alpha': aux, 'beta': np.array(best_beta_std), 'sessions_errors': best_dict_pairs_sessions_errors, 'error': best_ratio_test}
	return all_ratios_val, all_ratios_test, dict_best






########################################## EMPIRICAL BAYES ESTIMATORS ##########################################


### Empirical Bayes on the log_session_length / We allow to use the theoretical value of lambda 

def EB_log_towards_mean(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, number_points_users_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val=1, baseline_test=1, theoretical_alpha=False):
	all_EB_val_MAE            = []
	all_EB_test_MAE           = []
	all_means_users_train     = [np.mean(y_log_data_users_train[ indexes_listeners_train[i] ]) for i in range(len(indexes_listeners_train))]

	### Which alpha ? -> range of values or theoretical ?
	if theoretical_alpha:
		est_sigma0_2, est_sigma1_2 = 0, 0

		for i in range(len(indexes_listeners_train)):
			
			y_data_user_train = y_log_data_users_train[ indexes_listeners_train[i] ]
			n_user  = len(y_data_user_train)
			
			if n_user > 1:
				y_norm  = np.linalg.norm(y_data_user_train)**2
				y_sum   = np.sum(y_data_user_train)**2

				aux_sigma0_2  = (y_sum - y_norm) / float(n_user*(n_user-1))
				est_sigma0_2 += aux_sigma0_2
				est_sigma1_2 += y_norm/float(n_user) - aux_sigma0_2
				#print est_sigma0_2, est_sigma1_2
		
		list_llambda = [est_sigma1_2/est_sigma0_2]


	### Results
	best_ratio_val                  = 1e6
	best_ratio_test                 = 1e6
	best_llambda                    = 0
	dict_best_pairs_sessions_errors = {}



	for llambda in list_llambda:
		EB_val_MAE  = []
		EB_test_MAE = []
		dict_pairs_sessions_errors = {}

		for i in range(len(indexes_listeners_test)):  
		
		### Data user
			y_data_user_train = y_log_data_users_train[ indexes_listeners_train[i] ]
			mean_user_train   = np.mean(y_data_user_train)
			number_points_user_train = number_points_users_train[i]


		### Prediction
			EB_train       = mean_user_train / (1 + llambda/number_points_user_train)
			EB_train       = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*EB_train
			
			y_data_user_val =  y_data_users_val[ indexes_listeners_val[i] ]
			sum_error_user  = [abs(math.exp(EB_train) - y_data)  for y_data in y_data_user_val]
			EB_val_MAE.extend(sum_error_user)

			y_data_user_test =  y_data_users_test[ indexes_listeners_test[i] ]
			sum_error_user   =  [abs(math.exp(EB_train) - y_data) for y_data in y_data_user_test]
			EB_test_MAE.extend(sum_error_user)


			n_user_train = len(y_log_data_users_train[ indexes_listeners_train[i] ])
			if n_user_train not in dict_pairs_sessions_errors.keys():
				dict_pairs_sessions_errors[n_user_train] = list(sum_error_user) 
			else:
				dict_pairs_sessions_errors[n_user_train].extend( sum_error_user ) 


		EB_val_MAE  = np.median(EB_val_MAE)
		EB_test_MAE = np.median(EB_test_MAE)


		EB_val_MAE  /= baseline_val
		EB_test_MAE /= baseline_test
		if EB_test_MAE < best_ratio_val:
			best_ratio_val  = EB_val_MAE

			best_ratio_test = EB_test_MAE
			best_llambda    = llambda
			dict_best_pairs_sessions_errors = dict_pairs_sessions_errors

		all_EB_val_MAE.append(EB_val_MAE)
		all_EB_test_MAE.append(EB_test_MAE)

	dict_best = {'llambda': best_llambda, 'error': EB_test_MAE, 'sessions_errors': dict_best_pairs_sessions_errors}
	return all_EB_val_MAE, all_EB_test_MAE, dict_best









########################################## ALTERNATIVE MINIMIZATION ##########################################


### Run the alternative_minimization procedure on a grid of parameters


def loop_alternative_minimization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, dict_baseline_pairs_sessions_errors, f, baseline_val=1, baseline_test=1, penalization='l2', use_L0=False, range_K=[0], is_test_standardized=False):

	write_and_print('\n\n###### REGULARIZATION '+penalization, f)

	### Parameters for Ridge
	dict_alpha_max = {'l2':power_method(x_data_users_train), 'l1':np.max(np.abs(np.dot(x_data_users_train.T, y_log_data_users_train)))}
	alpha_max      = dict_alpha_max[penalization]
	list_alpha 	   = [alpha_max*0.7**i for i in range(N_ALPHA)][25:]

	av_time = 0
	size_grid = len(list_alpha)*len(list_llambda)


	### SVD for Ridge
	XTX         = np.dot(x_data_users_train.T, x_data_users_train)
	U, Diag, _  = np.linalg.svd(XTX)
	mu_max      = np.max(Diag)



	### Results
	all_ratios_val   = []
	all_ratios_test  = []
	best_ratio_val   = 1e6
	best_ratio_test  = 1e6

	best_beta_std, best_params      = [], (0,0)
	dict_best_pairs_sessions_errors = {}


	### Main loop
	old_beta_train_std            = np.zeros(np.array(x_data_users_train).shape[1])
	#all_means_x_data_users_train  = np.array([ params_train_df[col]['mean'] for col in x_train_columns])
	ones_N = np.ones(x_data_users_train.shape[0])


	n_llambda = len(list_llambda) 



	for alpha_reg in list_alpha:
		ratio_val_MAE  = []
		ratio_test_MAE = []
		write_and_print('\n####### Alpha: '+str(round(alpha_reg,4))+' #######', f)

		start_time = time.time()

		for jj in range(n_llambda):
			llambda = list_llambda[jj]


######################## TRY L0 ON XGBOOST FEATURES ########################
			
			for K in range_K: ### TRY: - decrease with ridge - so far, not distinction between K

			### Compute the estimator for alpha and llambda
				beta_train_std, EB_residual_users, number_loops = alternative_minimization(indexes_listeners_train, x_data_users_train, XTX, mu_max, U, Diag, y_log_data_users_train, number_points_user_train, alpha_reg, llambda, old_beta_train_std, f, penalization=penalization, use_L0=use_L0, K=K)

				if jj == 0: old_first_beta_train = np.copy(beta_train_std)
				old_beta_train_std = np.copy(beta_train_std) if jj<n_llambda-1 else old_first_beta_train
				
				


			### Compute the test accuracy ->is the data standardized ??
				if not is_test_standardized:
					beta_train = beta_train_std * np.array([(params_train_df['log_session_duration']['std']/params_train_df[col]['std']) for col in x_train_columns])


				start_time_pred = time.time()

				x_val_beta  = np.dot(x_data_users_val,  beta_train_std)
				x_test_beta = np.dot(x_data_users_test, beta_train_std)

			### Results
				ratio_cov_EB_val_MAE       = []
				ratio_cov_EB_test_MAE      = []
				dict_pairs_sessions_errors = {}

	 
				for i in range(len(indexes_listeners_test)): 

				######### VAL #######
					y_data_user_val = y_data_users_val[ indexes_listeners_val[i] ]
					n_user_val      = len(y_data_user_val)

					if n_user_val>0:	
						if not is_test_standardized:
							pred_EB_val     = params_train_df['log_session_duration']['std'] * EB_residual_users[i] + params_train_df['log_session_duration']['mean'] - np.dot(all_means_x_data_users_train, beta_train) 
							sum_error_user  = np.abs( np.exp(np.dot(x_data_user_val, beta_train) + pred_EB_val*ones_N[:n_user_val] ) - y_data_user_val )
						else:
							pred_EB_val     = params_train_df['log_session_duration']['std'] * (EB_residual_users[i] + x_val_beta[indexes_listeners_val[i] ]  ) + params_train_df['log_session_duration']['mean'] 
							sum_error_user  = np.abs( np.exp(pred_EB_val) - y_data_user_val )
					else:
						sum_error_user = []

					ratio_cov_EB_val_MAE.extend( sum_error_user )


				######### TEST #######
					y_data_user_test = y_data_users_test[ indexes_listeners_test[i] ]
					n_user_test      = len(y_data_user_test)
					
					if n_user_test>0:
						
						if not is_test_standardized:
							cov_EB_test     = params_train_df['log_session_duration']['std'] * EB_residual_users[i] + params_train_df['log_session_duration']['mean'] - np.dot(all_means_x_data_users_train, beta_train) 
							sum_error_user  = np.abs( np.exp(np.dot(x_data_user_test, beta_train) + cov_EB_test*ones_N[:n_user_test] ) - y_data_user_test )
						else:
							pred_EB_test    = params_train_df['log_session_duration']['std'] * (EB_residual_users[i] + x_test_beta[indexes_listeners_test[i]] ) + params_train_df['log_session_duration']['mean'] 
							sum_error_user  = np.abs( np.exp(pred_EB_test) - y_data_user_test )
					else:
						sum_error_user = []

					n_user_train = len(y_log_data_users_train[ indexes_listeners_train[i] ])
					if n_user_train not in dict_pairs_sessions_errors.keys():
						dict_pairs_sessions_errors[n_user_train] = list(sum_error_user) 
					else:
						dict_pairs_sessions_errors[n_user_train].extend( sum_error_user ) 
							
					ratio_cov_EB_test_MAE.extend( sum_error_user )

				ratio_cov_EB_val_MAE  = np.median(ratio_cov_EB_val_MAE)
				ratio_cov_EB_test_MAE = np.median(ratio_cov_EB_test_MAE)

				ratio_cov_EB_val_MAE  /= baseline_val
				ratio_cov_EB_test_MAE /= baseline_test

				av_time += (time.time() - start_time_pred)/size_grid


				if not use_L0:
					write_and_print('Lambda: '+str(round(llambda,2))+' Val prediction: '+str(ratio_cov_EB_val_MAE)+' Test prediction: '+str(ratio_cov_EB_test_MAE) , f)
				else:
					write_and_print('Lambda: '+str(round(llambda,2))+' K:'+str(K)+' Val prediction: '+str(ratio_cov_EB_val_MAE)+' Test prediction: '+str(ratio_cov_EB_test_MAE)+'\n', f)

				if ratio_cov_EB_val_MAE < best_ratio_val:
					best_ratio_val  = ratio_cov_EB_val_MAE

					best_ratio_test = ratio_cov_EB_test_MAE
					best_beta_std   = np.copy(beta_train_std)
					best_params     = alpha_reg, llambda

					dict_best_pairs_sessions_errors = dict_pairs_sessions_errors

			ratio_val_MAE.append( ratio_cov_EB_val_MAE)
			ratio_test_MAE.append(ratio_cov_EB_test_MAE)

		all_ratios_val.append(ratio_val_MAE)
		all_ratios_test.append(ratio_test_MAE)

		write_and_print('Time loop: '+str(time.time() - start_time)+'\n', f)

	write_and_print('\nAVERAGE TIME PREDICTION: '+str(av_time)+'\n', f)
	dict_best = {'av_time_prediction':av_time, 'alpha': best_params[0], 'lambda': best_params[1], 'beta': np.array(best_beta_std), 'error':best_ratio_test, 'sessions_errors': dict_best_pairs_sessions_errors}
	return all_ratios_val, all_ratios_test, dict_best





### Alternative minimizaiton procedure for Empirical Bayes with covariates: we alternate 2 steps
		# - one EB as above
		# - one ridge regression on the residuals


def alternative_minimization(indexes_listeners_train, X_train, XTX, mu_max, U, Diag, y_train, number_points_user, alpha_reg, llambda, beta_init, f, penalization='l2', use_L0=False, K=0):
	old_obj_val   =  1e6
	obj_val       = -1e6
	N, P = X_train.shape

	
	beta_train    = beta_init
	ridge_target  = y_train - np.dot(X_train, beta_train)
	XTy = np.dot(X_train.T, ridge_target)
	

	### Stopping criterion
	number_loops = 0
	while (old_obj_val- obj_val) / old_obj_val > 1e-2:
		number_loops += 1
	
	### STEP 1: EB
		EB_residuals = []
		aux = 0

		for i in range(len(indexes_listeners_train)):  
			y_data_user        = y_train[ indexes_listeners_train[i] ]
			x_data_user        = X_train[ indexes_listeners_train[i] ]

			EB_residual_user  = np.mean( np.array(y_data_user-np.dot(x_data_user, beta_train)) ) / float(1 + llambda/float(number_points_user[i]))
			EB_residuals.append(EB_residual_user)

			for idx in indexes_listeners_train[i]: ridge_target[idx] = y_train[idx] - EB_residual_user
				
				
	## STEP 2: REGRESSION   
		if use_L0:
			if K==P and penalization == 'l2':
				beta_train = np.dot(np.dot(np.dot(U, np.diag(1/(Diag + alpha_reg))), U.T), np.dot(X_train.T, ridge_target))
			else:
				beta_train = DFO_nlarge('l2', X_train, ridge_target, alpha_reg, XTX=XTX, XTy=np.dot(X_train.T, ridge_target), mu_max=mu_max, beta_start=beta_init, use_L0=True, K=K)
			

		else:
			if penalization == 'l2':
				#estimator  = Ridge(alpha=alpha_reg, fit_intercept=False, solver='svd')
				beta_train = np.dot(np.dot(np.dot(U, np.diag(1/(Diag + alpha_reg))), U.T), np.dot(X_train.T, ridge_target))

			if penalization == 'l1':
				#estimator  = Lasso(alpha=alpha_reg/float(N), fit_intercept=False, max_iter=10000)
				#estimator.fit(X_train, ridge_target)
				#beta_train = np.copy(estimator.coef_)
				beta_train = DFO_nlarge('l1', X_train, ridge_target, alpha_reg, XTX=XTX, XTy=np.dot(X_train.T, ridge_target), mu_max=mu_max, beta_start=beta_init, use_L0=False)
				


		old_obj_val = obj_val
		dict_reg    = {'l1':np.sum(np.abs(beta_train)), 'l2':np.linalg.norm(beta_train)**2}    
		obj_val     = np.linalg.norm( ridge_target - np.dot(X_train, beta_train))**2  + llambda*np.linalg.norm(EB_residuals)**2 + alpha_reg*dict_reg[penalization]
		#print 'Objval: '+str(obj_val), (old_obj_val- obj_val) / old_obj_val

	write_and_print('Number loops: '+str(number_loops), f)
	write_and_print('Size support: '+str( (len(np.where(beta_train!=0)[0]), len(beta_train)) ), f)
	return beta_train, EB_residuals, number_loops





########################################## ALTERNATIVE MINIMIZATION WITH XGBOOST ##########################################


def loop_alternative_minimization_xgboost(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, f, baseline_val=1, baseline_test=1, only_size_2=False):

# INPUT
# x_data_users_test_STD: carefull standardization
# y_data_users_test    : for test accuracy

	print list_llambda
	max_depth    = [6, 10] if not only_size_2 else [2]
	n_estimators = [50, 100] 
	#n_estimators = [100] 

	
	### Parameters 
	all_ratios_val   = []
	all_ratios_test  = []

	best_ratio_val  = 1e6
	best_ratio_test = 1e6

	test_params = {
		'learning_rate': [1e-1],
		'max_depth': max_depth,
		'n_estimators': n_estimators,
		'subsample': [1],
		'colsample_bytree': [1]
	}

	params_grid = ParameterGrid(test_params)
	dict_best_pairs_sessions_errors = {}


	av_time = 0
	size_grid = len([p for p in params_grid])*len(list_llambda)

	ones_N = np.ones(x_data_users_train.shape[0])

	

	for params in params_grid:
		ratio_val_MAE  = []
		ratio_test_MAE = []
		write_and_print('\nParams: '+str(params), f)
		
		start_time = time.time()

		for llambda in list_llambda:

		### Compute the estimator for alpha and llambda
			xgb_model, EB_residual_users = alternative_minimization_xgboost(indexes_listeners_train, x_data_users_train, y_log_data_users_train, number_points_user_train, params, llambda, f)

		### Compute the test accuracy -> the data is standardized !!
			ratio_cov_EB_val_MAE  = []
			ratio_cov_EB_test_MAE = []
			dict_pairs_sessions_errors = {} 


			start_time_pred =  time.time()
			xgb_all_user_predictions_val  = xgb_model.predict(x_data_users_val)
			xgb_all_user_predictions_test = xgb_model.predict(x_data_users_test)
			#print time.time() - start_time_pred

			for i in range(len(indexes_listeners_test)): 

			######### VAL #######
				#x_data_user_val = x_data_users_val[ indexes_listeners_val[i] ]
				y_data_user_val = y_data_users_val[ indexes_listeners_val[i] ]
				n_user_val      = len(y_data_user_val)

				if n_user_val>0:
					xgb_val_predictions   = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*(xgb_all_user_predictions_val[ indexes_listeners_val[i] ] + EB_residual_users[i]) 
					sum_error_user        = np.abs( np.exp(xgb_val_predictions) - y_data_user_val ) 
				else:
					sum_error_user = []
				ratio_cov_EB_val_MAE.extend( sum_error_user )


			######### TEST #######
				#x_data_user_test = x_data_users_test[ indexes_listeners_test[i] ]
				y_data_user_test = y_data_users_test[ indexes_listeners_test[i] ]
				n_user_test      = len(y_data_user_test)

				if n_user_test>0:
					xgb_test_predictions   = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*(xgb_all_user_predictions_test[ indexes_listeners_test[i] ] + EB_residual_users[i]) 
					sum_error_user 		   = np.abs( np.exp(xgb_test_predictions) - y_data_user_test )
				else:
					sum_error_user = []

				
				n_user_train = len(y_log_data_users_train[ indexes_listeners_train[i] ])
				if n_user_train not in dict_pairs_sessions_errors.keys():
					dict_pairs_sessions_errors[n_user_train] = list(sum_error_user) 
				else:
					dict_pairs_sessions_errors[n_user_train].extend( sum_error_user ) 


				ratio_cov_EB_test_MAE.extend( sum_error_user )

				
			ratio_cov_EB_val_MAE  = np.median(ratio_cov_EB_val_MAE)
			ratio_cov_EB_test_MAE = np.median(ratio_cov_EB_test_MAE)

			ratio_cov_EB_val_MAE  /= baseline_val
			ratio_cov_EB_test_MAE /= baseline_test

			av_time += (time.time() - start_time_pred)/size_grid


			write_and_print('Lambda: '+str(round(llambda))+' Val prediction: '+str(round(ratio_cov_EB_val_MAE, 3))+' Test prediction: '+str(round(ratio_cov_EB_test_MAE, 3)) , f)
			if ratio_cov_EB_val_MAE < best_ratio_val:
				best_ratio_val  = ratio_cov_EB_val_MAE

				best_model      = xgb_model._Booster
				best_ratio_test = ratio_cov_EB_test_MAE
				best_params     = params

				dict_best_pairs_sessions_errors = dict_pairs_sessions_errors


			ratio_val_MAE.append( ratio_cov_EB_val_MAE)
			ratio_test_MAE.append(ratio_cov_EB_test_MAE)
		
		all_ratios_val.append(ratio_val_MAE)
		all_ratios_test.append(ratio_test_MAE)
		write_and_print('Time loop: '+str(time.time() - start_time), f)


	write_and_print('\nAVERAGE TIME PREDICTION: '+str(av_time)+'\n', f)
	dict_best = {'av_time_prediction':av_time, 'params': best_params, 'model': best_model, 'error': best_ratio_test, 'sessions_errors': dict_best_pairs_sessions_errors}
	return all_ratios_val, all_ratios_test, dict_best





def alternative_minimization_xgboost(indexes_listeners_train, x_data_users, y_data_users, number_points_user, params, llambda, f):
	old_obj_val   =  1e6
	obj_val       = -1e6
	ridge_target  = np.copy(y_data_users)


	### Stopping criterion
	start_time = time.time()
	number_loops = -1
	while (old_obj_val- obj_val) / old_obj_val > 1e-2:
		number_loops += 1

		xgb_model = XGBRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], 
								 subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
		xgb_model.fit(x_data_users, ridge_target)

		EB_residuals = []
		aux = 0

		for i in range(len(indexes_listeners_train)):  
			y_data_user        = y_data_users[ indexes_listeners_train[i] ]
			x_data_user        = x_data_users[ indexes_listeners_train[i] ]

			EB_residual_user  = np.mean( np.array(y_data_user) - xgb_model.predict(np.array(x_data_user)) ) / float(1 + llambda/float( len(y_data_user) ))
			EB_residuals.append(EB_residual_user)

			for idx in indexes_listeners_train[i]: ridge_target[idx] = y_data_users[idx] - EB_residual_user


		old_obj_val    = obj_val
		obj_val        = np.linalg.norm( ridge_target - xgb_model.predict(x_data_users) )**2 + llambda*np.linalg.norm(EB_residuals)**2
		#print 'Objval: '+str(obj_val), (old_obj_val- obj_val) / old_obj_val
	
	write_and_print('Number loops: '+str(number_loops), f)
	write_and_print('Time alt-min: '+str(time.time() - start_time), f)

	return xgb_model, EB_residuals





########################################## XGBOOST ##########################################


def xgboost(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, y_data_users_train, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, params_train_df, f, baseline_val=1, baseline_test=1, predict_log=False, only_size_2=False):

	max_depth    = [6, 10] if not only_size_2 else [2]
	n_estimators = [50, 100] if not only_size_2 else [50]

	test_params = {
		'learning_rate': [5e-2, 1e-1],
		'max_depth': max_depth,
		'n_estimators': n_estimators,
		'subsample': [1],
		'colsample_bytree': [1]
	}


	params_grid     = ParameterGrid(test_params)
	all_ratios_val  = [] 
	all_ratios_test = [] 
	all_models_str  = []
	all_dict_pairs_sessions_errors = []

	av_time = 0
	size_grid = len([p for p in params_grid])

	#y_data_users_train = params_train_df['session_duration']['std'] * np.array(y_data_users_train) + params_train_df['session_duration']['mean'] 

	for params in params_grid:
		dict_pairs_sessions_errors = {}
		print params

		start_time = time.time()
		xgb_model = XGBRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], 
								 subsample=params['subsample'], colsample_bytree=params['colsample_bytree'])
		
		xgb_model.fit(x_data_users_train, y_data_users_train)	


	### If trees of size 2, we keep a description of the best model
		all_models_str.append( xgb_model._Booster )

		start_time = time.time()

	### Data not standardized
		xgb_all_user_predictions_val  = xgb_model.predict(x_data_users_val_STD)
		xgb_all_user_predictions_test = xgb_model.predict(x_data_users_test_STD)

		
		if not predict_log:
			xgb_all_user_val_MAE          = np.median(np.abs(xgb_all_user_predictions_val  - y_data_users_val))
			xgb_all_user_test_MAE         = np.median(np.abs(xgb_all_user_predictions_test - y_data_users_test))
		
		else:
			xgb_all_user_val_predictions = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*xgb_all_user_predictions_val
			xgb_all_user_val_MAE         = np.median(np.abs( np.exp(xgb_all_user_val_predictions)  - y_data_users_val))

			xgb_all_user_test_predictions = params_train_df['log_session_duration']['mean'] + params_train_df['log_session_duration']['std']*xgb_all_user_predictions_test
			xgb_all_user_test_MAE         = np.median(np.abs(np.exp(xgb_all_user_test_predictions) - y_data_users_test))




	######### ADD FOR PLOT #######
		dict_pairs_sessions_errors = {} 
		for i in range(len(indexes_listeners_test)): 

			y_data_user_test = y_data_users_test[ indexes_listeners_test[i] ]
			n_user_test      = len(y_data_user_test)

			if n_user_test>0:
				sum_error_user = np.abs( xgb_all_user_predictions_test[ indexes_listeners_test[i] ] - y_data_user_test ) 
			else:
				sum_error_user = []

			n_user_train = len(y_data_users_train[ indexes_listeners_train[i] ])
			if n_user_train not in dict_pairs_sessions_errors.keys():
				dict_pairs_sessions_errors[n_user_train] = list(sum_error_user)
			else:
				dict_pairs_sessions_errors[n_user_train].extend( sum_error_user ) 

		all_dict_pairs_sessions_errors.append(dict_pairs_sessions_errors)

		all_ratios_val.append(xgb_all_user_val_MAE/baseline_val)
		all_ratios_test.append(xgb_all_user_test_MAE/baseline_test)

		av_time += (time.time() - start_time)/size_grid
		print time.time() - start_time

		write_and_print('Time xgboost: '+str(time.time() - start_time), f)
		write_and_print('Val error : '+str(xgb_all_user_val_MAE/baseline_val), f)
		write_and_print('Test error: '+str(xgb_all_user_test_MAE/baseline_test)+'\n', f)


	write_and_print('\nAVERAGE TIME PREDICTION: '+str(av_time)+'\n', f)
	
	argmin    = np.argmin(all_ratios_val)    
	dict_best = {'av_time_prediction':av_time, 'params': params_grid[argmin], 'model': all_models_str[argmin], 'error': all_ratios_test[argmin], 'sessions_errors': all_dict_pairs_sessions_errors[argmin]}
	return all_ratios_val, all_ratios_test, dict_best





########################################## HELPER ##########################################


def power_method(X):
	P = X.shape[1]

#---Compute the highest eigenvector
	highest_eigvctr     = np.random.rand(P) #random intialization
	old_highest_eigvctr = 1e6*np.ones(P)
	
	while np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2:   #CV criterion
		old_highest_eigvctr = highest_eigvctr
		highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))   #p is large
		highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
	
#---Deduce the highest eigenvalue
	X_highest_eigval = np.dot(X, highest_eigvctr)
	highest_eigval   = np.dot(X_highest_eigval.T, X_highest_eigval)/np.linalg.norm(highest_eigvctr)
	
	return highest_eigval






def write_and_print(text,f):
	print text
	f.write('\n'+text)











