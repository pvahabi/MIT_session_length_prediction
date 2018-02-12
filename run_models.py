import pandas as pd
import numpy as np
import math
import time

from models import *




########################################## BASELINE ##########################################

def run_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test, f, output_file=''):

	start_time = time.time()
	baseline_val_MAE, baseline_test_MAE, dict_baseline_pairs_sessions_errors  = baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_data_users_train, params_train_df, y_data_users_val, y_data_users_test)
	write_and_print('\n\nBASELINE VAL : '+str(round(baseline_val_MAE/baseline_val_MAE,3)) , f)
	write_and_print('BASELINE TEST: '+str(round(baseline_test_MAE/baseline_test_MAE,3)) , f)
	write_and_print('BASELINE TIME: '+str(time.time()-start_time) , f)


	if len(output_file)>0:
		g = open(output_file+'/dict_baseline.txt',"w")
		g.write( str(dict_baseline_pairs_sessions_errors) )
		g.close()

	return baseline_val_MAE, baseline_test_MAE, dict_baseline_pairs_sessions_errors





########################################## LOG BASELINE ##########################################


def run_log_baseline(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, f):

	start_time = time.time()
	log_baseline_val_MAE, log_baseline_test_MAE, log_baseline_pairs_sessions_errors = baseline_log(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors)
	write_and_print('\n\nLOG BASELINE VAL : '+str(round(log_baseline_val_MAE/baseline_val_MAE,3)), f)
	write_and_print('LOG BASELINE TEST: '+str(round(log_baseline_test_MAE/baseline_test_MAE,3)), f)
	write_and_print('LOG BASELINE TIME: '+str(time.time()-start_time) , f)



########################################## RIDGE ##########################################


def run_ridge(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, params_train_df, x_data_users_val, y_data_users_val, x_data_users_test, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, theoretical_alpha=False):

	start_time = time.time()
	all_ratios_val, all_ratios_test, dict_best_params_ridge = ridge(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, params_train_df, x_data_users_val, y_data_users_val, x_data_users_test, y_data_users_test, dict_baseline_pairs_sessions_errors, f, baseline_val=baseline_test_MAE, baseline_test=baseline_test_MAE, theoretical_alpha=theoretical_alpha, is_test_standardized=True)

	print all_ratios_val, all_ratios_test

	dict_word1 = {True:'theoretical_ridge', False:'ridge'}
	dict_word2 = {True:'THEORETICAL RIDGE', False:'RIDGE'}

	np.save(output_file+'/all_ratios_'+dict_word1[theoretical_alpha]+'_VAL',  all_ratios_val)
	np.save(output_file+'/all_ratios_'+dict_word1[theoretical_alpha]+'_TEST', all_ratios_test)

	np.save(output_file+'/'+dict_word1[theoretical_alpha]+'_beta', dict_best_params_ridge['beta'])
	np.save(output_file+'/columns_beta', x_train_columns)
	
	write_and_print('\n\nBest    VAL  MAE '+dict_word2[theoretical_alpha]+': '+str(round(np.min(all_ratios_val), 3)), f)
	write_and_print('Best    TEST MAE '+dict_word2[theoretical_alpha]+': '+str(round(np.min(all_ratios_test), 3)), f)
	write_and_print('Correct TEST MAE '+dict_word2[theoretical_alpha]+': '+str(round(dict_best_params_ridge['error'], 3)), f)
	write_and_print('TIME '+dict_word2[theoretical_alpha]+': '+str(time.time()-start_time)+'\n\n' , f)

###### dict best params
	g = open(output_file+'/'+dict_word1[theoretical_alpha]+'.txt',"w")
	g.write( str(dict_best_params_ridge) )
	g.close()

###### dict best params
	g = open(output_file+'/dict_'+dict_word1[theoretical_alpha]+'.txt',"w")
	g.write( str(dict_best_params_ridge['sessions_errors']) )
	g.close()






########################################## EMPIRICAL BAYES ##########################################


def run_EB_log(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, number_points_user_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, theoretical_alpha=False):

	start_time = time.time()
	list_llambda = np.arange(0,8,.2)
	all_ratios_EB_val_MAE, all_ratios_EB_test_MAE, dict_best_params_EB = EB_log_towards_mean(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, y_log_data_users_train, number_points_user_train, params_train_df, y_data_users_val, y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val=baseline_val_MAE, baseline_test=baseline_test_MAE, theoretical_alpha=theoretical_alpha)
	
	print all_ratios_EB_val_MAE, all_ratios_EB_test_MAE

	dict_word1 = {True:'theoretical_EB', False:'EB'}
	dict_word2 = {True:'THEORETICAL EMPIRICAL BAYES', False:'EMPIRICAL BAYES'}

	np.save(output_file+'/all_ratios_'+dict_word1[theoretical_alpha]+'_with_log_VAL',  all_ratios_EB_val_MAE)
	np.save(output_file+'/all_ratios_'+dict_word1[theoretical_alpha]+'_with_log_TEST', all_ratios_EB_test_MAE)
	
	
	write_and_print('\n\nBest  VAL MAE  '+dict_word2[theoretical_alpha]+' with LOG   : '+str(round(np.min(all_ratios_EB_val_MAE), 3)), f)
	write_and_print('Best TEST MAE  '+dict_word2[theoretical_alpha]+' with LOG   : '+str(round(np.min(all_ratios_EB_test_MAE), 3)), f)

	write_and_print('Lambda  '+dict_word2[theoretical_alpha]+' with LOG: '+str(dict_best_params_EB['llambda']), f)
	write_and_print('Correct TEST MAE '+dict_word2[theoretical_alpha]+': '+str(round(dict_best_params_EB['error'], 3)), f)
	
	write_and_print('TIME '+dict_word2[theoretical_alpha]+': '+str(time.time()-start_time) , f)

###### dict best params
	g = open(output_file+'/'+dict_word1[theoretical_alpha]+'.txt',"w")
	g.write( str(dict_best_params_EB) )
	g.close()

###### dict best params
	g = open(output_file+'/dict_'+dict_word1[theoretical_alpha]+'.txt',"w")
	g.write( str(dict_best_params_EB['sessions_errors']) )
	g.close()







########################################## XGBOOST ##########################################


def run_xgboost(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, y_data_users_train, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, params_train_df, baseline_val_MAE, baseline_test_MAE, output_file, f, predict_log=False):

	start_time = time.time()
	#if not predict_log:
	all_ratios_val, all_ratios_test, dict_best_params_xgboost = xgboost(indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, y_data_users_train, x_data_users_val_STD,  y_data_users_val, x_data_users_test_STD,  y_data_users_test, params_train_df, f, baseline_val=baseline_val_MAE, baseline_test=baseline_test_MAE, predict_log=predict_log)
	

	dict_word1 = {True:'xgboost_log', False:'xgboost'}
	dict_word2 = {True:'XGBOOST LOG', False:'XGBOOST'}

	np.save(output_file+'/all_ratios_'+dict_word1[predict_log]+'_VAL',  all_ratios_val)
	np.save(output_file+'/all_ratios_'+dict_word1[predict_log]+'_TEST', all_ratios_test)
	
	write_and_print('\nBest  VAL MAE  '+dict_word2[predict_log]+': '+str(round(np.min(all_ratios_val), 3)), f)
	write_and_print('Best TEST MAE  '+dict_word2[predict_log]+': '+str(round(np.min(all_ratios_test), 3)), f)

	write_and_print('\nParams '+dict_word2[predict_log]+': '+str(dict_best_params_xgboost['params']), f)
	write_and_print('Correct TEST MAE '+dict_word2[predict_log]+': '+str(round(dict_best_params_xgboost['error'], 3)), f)
	write_and_print('TIME '+dict_word2[predict_log]+': '+str(time.time()-start_time) , f)

###### dict best params
	g = open(output_file+'/dict_'+dict_word1[predict_log]+'.txt',"w")
	g.write( str(dict_best_params_xgboost['sessions_errors']) )
	g.close()







########################################## EMPIRICAL BAYES WITH RIDGE / LASSO ##########################################


def run_EB_penalization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f, penalization='l2', use_L0=False, range_K=[0]):

	start_time = time.time()
	all_ratios_val, all_ratios_test, dict_best_params_alt_min_penalization = loop_alternative_minimization(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, dict_baseline_pairs_sessions_errors, f, baseline_val=baseline_val_MAE, baseline_test=baseline_test_MAE, penalization=penalization, use_L0=use_L0, range_K=range_K, is_test_standardized=True)

	dict_word1 = {'l1':'lasso', 'l2':'ridge'} if not use_L0 else {'l1':'lasso_L0', 'l2':'ridge_L0'}
	dict_word2 = {'l1':'LASSO', 'l2':'RIDGE'} if not use_L0 else {'l1':'LASSO L0', 'l2':'RIDGE L0'}

	np.save(output_file+'/all_ratios_alternative_'+dict_word1[penalization]+'_VAL', all_ratios_val)
	np.save(output_file+'/all_ratios_alternative_'+dict_word1[penalization]+'_TEST', all_ratios_test)

	np.save(output_file+'/alternative_beta_'+dict_word1[penalization], dict_best_params_alt_min_penalization['beta'])
	np.save(output_file+'/columns_beta', x_train_columns)

	write_and_print('\nBest VAL  MAE ALT_MIN '+dict_word2[penalization]+': '+str(round(np.min(all_ratios_val), 3)), f)
	write_and_print('Best TEST MAE ALT_MIN '+dict_word2[penalization]+': '+str(round(np.min(all_ratios_test), 3)), f)

	write_and_print('\nAlpha ALT_MIN: '+str(dict_best_params_alt_min_penalization['alpha'])+' Lambda ALT_MIN: '+str(dict_best_params_alt_min_penalization['lambda']), f)
	write_and_print('Correct MAE ALT_MIN '+dict_word2[penalization]+': '+str(round(dict_best_params_alt_min_penalization['error'], 3)), f)
	write_and_print('TIME '+dict_word2[penalization]+': '+str(time.time()-start_time) , f)


###### dict best params
	g = open(output_file+'/alternative_'+dict_word1[penalization]+'.txt',"w")
	g.write( str(dict_best_params_alt_min_penalization) )
	g.close()

###### dict best params
	g = open(output_file+'/dict_'+dict_word1[penalization]+'.txt',"w")
	g.write( str(dict_best_params_alt_min_penalization['sessions_errors']) )
	g.close()




########################################## EMPIRICAL BAYES WITH XGBOOST ##########################################

## CAREFULL: test standardized

def run_EB_xgboost(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, dict_baseline_pairs_sessions_errors, baseline_val_MAE, baseline_test_MAE, output_file, f):

	start_time = time.time()
	all_ratios_val, all_ratios_test, dict_best_params_alt_min_xgboost = loop_alternative_minimization_xgboost(list_llambda, indexes_listeners_train, indexes_listeners_val, indexes_listeners_test, x_data_users_train, x_train_columns, y_log_data_users_train, number_points_user_train, params_train_df, x_data_users_val,  y_data_users_val, x_data_users_test,  y_data_users_test, f, baseline_val=baseline_val_MAE, baseline_test=baseline_test_MAE)
	
	np.save(output_file+'/all_ratios_alternative_xgboost_VAL',  all_ratios_val)
	np.save(output_file+'/all_ratios_alternative_xgboost_TEST', all_ratios_test)

	dict_best_params_alt_min_xgboost['model'].dump_model(output_file+'/best_model_alternative_xgboost.txt')

	write_and_print('\nBest VAL MAE ALT_MIN XGBOOST: '+str(round(np.min(all_ratios_val), 3)), f)
	write_and_print('Best TEST MAE ALT_MIN XGBOOST: '+str(round( np.min(all_ratios_test), 3)), f)

	write_and_print('\nParams XGBOOST ALT_MIN: '+str(dict_best_params_alt_min_xgboost['params']), f)
	write_and_print('Correct TEST MAE ALT_MIN XGBOOST: '+str(round(dict_best_params_alt_min_xgboost['error'], 3)), f)
	write_and_print('TIME ALT_MIN XGBOOST: '+str(time.time()-start_time) , f)


###### dict best params
	g = open(output_file+'/alternative_xgboost.txt',"w")
	g.write( str(dict_best_params_alt_min_xgboost) )
	g.close()


###### dict best params
	g = open(output_file+'/dict_alternative_xgboost.txt',"w")
	g.write( str(dict_best_params_alt_min_xgboost['sessions_errors']) )
	g.close()
	







