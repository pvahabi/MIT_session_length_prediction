import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt


def plot_column_importance(x_train_columns, beta_train):
	#columns_Xtrain = data_train.drop(['listener_id', 'session_duration', 'log_session_duration'], axis=1).columns

	if len(x_train_columns)!=beta_train.shape[0]:
		print 'NOT SAME LENGTH'
	else:
		print 'SAME LENGTH'

	order = np.argsort(np.abs(beta_train))[::-1]
	columns_ordered = x_train_columns[order]
	coefs_ordered   = np.abs(beta_train[order])

	for i in range(coefs_ordered.shape[0]):
		print columns_ordered[i], coefs_ordered[i]


	fig    = plt.figure(figsize=(15,5))
	ax     = plt.subplot()
	index  = np.arange(len(x_train_columns))
	width  = .8
	rects1 = ax.bar(index, coefs_ordered, width=width)

	ax.set_ylabel('Scores', size=16)
	plt.title('Coefficients by decreasing order', size=18)


	for tick in ax.get_xticklabels():
		tick.set_rotation(90)
	_ = ax.set_xticks(index )
	_ = ax.set_xticklabels(columns_ordered)






def plot_errors_by_sessions(dict_EB_pairs_sessions_errors, dict_best_params_ridge, dict_best_params_alt_min_ridge):
	fig = plt.figure(figsize=(15,5))
	ax1 = fig.add_subplot(1,1,1)

	plt.plot([-1,32], [1,1], c='k', label='Baseline')
	plt.plot(dict_EB_pairs_sessions_errors.keys()[:27], dict_EB_pairs_sessions_errors.values()[:27], c='g', label='Empirical Bayes')
	plt.plot(dict_best_params_ridge['sessions_errors'].keys()[:27], dict_best_params_ridge['sessions_errors'].values()[:27], c='r', label='Ridge')
	plt.plot(dict_best_params_alt_min_ridge['sessions_errors'].keys()[:27], dict_best_params_alt_min_ridge['sessions_errors'].values()[:27], c='b', label='EB-Ridge')

	ax1.set_xlim([-1,32])

	legend = ax1.legend(loc=4)
	for label in legend.get_texts(): label.set_fontsize('xx-large')
		

	ax1.set_title('Test errors by number of train sessions', fontsize=20,loc='center')
	ax1.set_xlabel('Number of sessions', fontsize=18)
	ax1.set_ylabel('Normalized MAE', fontsize=18)






def old_bars_test_errors(all_baseline_MAE, all_EB_MAE, all_EB_xgboost_MAE):
	

	plt.ioff() #no plot
	fig = plt.figure(figsize=(10,7))
	ax  = fig.add_subplot(1,1,1)
	
	colors ={0:'k', 1:'g', 2:'b', 3:'#FFA500', 4:'#FFA500', 5:'#FFA500', 6:'#FFA500'}
	

	width = .33
	rects = ax.bar(-width/2,  all_baseline_MAE[0],   width/2, color=colors[0], label='Baseline')        
	rects = ax.bar(0,         all_EB_MAE[0],         width/2, color=colors[1], label='Emp Bayes')        
	rects = ax.bar(+width/2,  all_EB_xgboost_MAE[0], width/2, color=colors[2], label='Best model')       

	for i in range(1, len(all_baseline_MAE)): 
		rects = ax.bar(i-width/2, all_baseline_MAE[i],   width/2, color=colors[0])        
		rects = ax.bar(i,         all_EB_MAE[i],         width/2, color=colors[1])        
		rects = ax.bar(i+width/2, all_EB_xgboost_MAE[i], width/2, color=colors[2]) 


#---Labels
	ax.set_xticks(range(len(all_baseline_MAE)))
	ax.set_xticklabels(['1-5', '5-20', '20+'])


	for ticks in [ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()]:
		for tick in ticks:
			tick.label.set_fontsize(16) 

	ax.set_ylim(bottom=0.5, top=1.15)

	legend = ax.legend(loc=2)
	for label in legend.get_texts():
		label.set_fontsize('xx-large')


	#ax.set_title('Test errors by number of sessions', fontsize=20,loc='center')
	ax.set_xlabel('Number of sessions per user', fontsize=18)
	ax.set_ylabel('nMAE', fontsize=18)










def bars_test_errors(all_baseline_MAE, all_EB_MAE, all_xgb_MAE, all_EB_xgboost_MAE):
	

	plt.ioff() #no plot
	fig = plt.figure(figsize=(10,7))
	ax  = fig.add_subplot(1,1,1)
	
	colors ={0:'k', 1:'g', 2:'b', 3:'#FFA500', 4:'#FFA500', 5:'#FFA500', 6:'#FFA500'}
	

	width = .4
	rects = ax.bar(-3*width/4,  all_baseline_MAE[0],   width/2, color=colors[0], label='Baseline')
	rects = ax.bar(  -width/4,  all_EB_MAE[0],         width/2, color=colors[1], label='Emp Bayes')        
	rects = ax.bar(   width/4,   all_xgb_MAE[0],        width/2, color=colors[2], label='Ridge')        
	rects = ax.bar(3*width/4,   all_EB_xgboost_MAE[0], width/2, color=colors[3], label='Best model')    
	   

	for i in range(1, len(all_baseline_MAE)): 
		rects = ax.bar(i -3*width/4,  all_baseline_MAE[i],   width/2, color=colors[0])
		rects = ax.bar(i   -width/4,  all_EB_MAE[i],         width/2, color=colors[1])    
		rects = ax.bar(i+   width/4,  all_xgb_MAE[i],        width/2, color=colors[2])        
		rects = ax.bar(i+ 3*width/4,   all_EB_xgboost_MAE[i], width/2, color=colors[3])


#---Labels
	ax.set_xticks(range(len(all_baseline_MAE)))
	ax.set_xticklabels(['1-10', '10-20', '20+'])


	for ticks in [ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()]:
		for tick in ticks:
			tick.label.set_fontsize(16) 

	ax.set_ylim(bottom=0.5, top=1.18)

	legend = ax.legend(loc=2)
	for label in legend.get_texts():
		label.set_fontsize('xx-large')


	#ax.set_title('Test errors by number of sessions', fontsize=20,loc='center')
	ax.set_xlabel('Number of sessions per user', fontsize=18)
	ax.set_ylabel('nMAE', fontsize=18)


