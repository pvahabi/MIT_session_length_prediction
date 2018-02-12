import sys
import os
import datetime
import random
import numpy as np

sys.path.append('../../graphics')
from plot import *

from collections import Counter

#from boxplot_averaged_test_errors import *



#name_dataset = 'small_pandora'
#name_dataset = 'large_pandora_mean'
name_dataset = 'last_fm_mean'


def bar_plots():

	number_points_user_train = np.load('../aux_plots/'+name_dataset+'/number_points_user_train.npy') 
	#dict = Counter(number_points_user_train)
	q1, q2 =  np.percentile(number_points_user_train, 10),  np.percentile(number_points_user_train, 20)
	print q1,q2

	dict_baseline_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_baseline.txt')
	dict_baseline = {}


	dict_EB_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_theoretical_EB.txt')
	dict_EB = {}


	dict_ridge_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_ridge.txt')
	dict_ridge = {}
	
	dict_xgb_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_xgboost.txt')
	#dict_xgb = {}

	#dict_xgb_txt = open('../aux_plots/'+name_dataset+'/basic/dict_xgboost.txt')
	dict_xgb = {}

	dict_EB_xgboost_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_alternative_xgboost.txt')
	dict_EB_xgboost = {}

	for (dict_txt, dict_)  in zip([dict_baseline_txt, dict_EB_txt, dict_ridge_txt, dict_xgb_txt, dict_EB_xgboost_txt], [dict_baseline, dict_EB, dict_ridge, dict_xgb, dict_EB_xgboost]):
		for dict_txt in dict_txt:
			dict_txt = dict_txt[1:-1].split(',')

			for points in dict_txt:
				points = points.split(':')
				n_user, MAE   = int(points[0]), float(points[1])
				dict_[n_user] = MAE


	SMALL = q1
	LARGE = q2

	#all_baseline_MAE = [[], [], []]
	#all_EB_MAE = [[], [], []]
	#all_EB_ridge = [[], [], []]
	#all_EB_xgb = [[[], [], []]
	#all_EB_xgboost_MAE = [[], [], []]


	for (all_MAE, dict_) in zip([all_baseline_MAE, all_EB_MAE, all_EB_ridge, all_EB_xgb, all_EB_xgboost_MAE], [dict_baseline, dict_EB, dict_ridge, dict_xgb, dict_EB_xgboost]):
		for n_train in dict_baseline.keys():
			if n_train <= SMALL:
				print dict_[n_train]
				all_MAE[0].extend( dict_[n_train] )

			if n_train <= LARGE:
				all_MAE[1].extend( dict_[n_train] )
			#elif n_train <= LARGE_BIS:

			if n_train > LARGE-1:
				all_MAE[2].extend( dict_[n_train] )
			#else:
			#	all_MAE[3] += dict_[n_train]


	copy_baseline = np.copy(all_baseline_MAE)
	for all_MAE in [all_baseline_MAE, all_EB_MAE, all_EB_ridge, all_EB_xgb, all_EB_xgboost_MAE]:
		for i in range(len(all_MAE)):
			all_MAE[i] /= copy_baseline[i]

	print np.round(all_EB_MAE,3), np.round(all_EB_ridge,3), np.round(all_EB_xgb,3),  np.round(all_EB_xgboost_MAE,3)


	bars_test_errors(all_baseline_MAE, all_EB_MAE, all_EB_xgb, all_EB_xgboost_MAE)
	plt.savefig('../aux_plots/'+name_dataset+'/advanced/bar_plots.pdf',bbox_inches='tight')
	plt.close()











def features_importance():

	print '##### BASIC FEATURES - RIDGE #####'
	x_train_columns = np.load('../aux_plots/'+name_dataset+'/basic/columns_beta.npy')
	beta_train      = np.load('../aux_plots/'+name_dataset+'/basic/ridge_beta.npy')

	plot_column_importance(x_train_columns, beta_train)
	plt.savefig('../aux_plots/'+name_dataset+'/basic/features_importance_ridge.pdf')
	plt.close()


	print '\n\n##### ADVANCED FEATURES - RIDGE #####'
	x_train_columns = np.load('../aux_plots/'+name_dataset+'/advanced/columns_beta.npy')
	beta_train      = np.load('../aux_plots/'+name_dataset+'/advanced/ridge_beta.npy')

	plot_column_importance(x_train_columns, beta_train)
	plt.savefig('../aux_plots/'+name_dataset+'/advanced/features_importance_ridge.pdf')
	plt.close()



	print '\n\n##### ADVANCED FEATURES - RIDGE #####'
	beta_train = np.load('../aux_plots/'+name_dataset+'/advanced/alternative_beta_ridge.npy')
	#beta_train = np.load('../aux_plots/advanced/alternative_beta_lasso.npy')
	plot_column_importance(x_train_columns, beta_train)
	plt.savefig('../aux_plots/'+name_dataset+'/advanced/features_importance_alternative_ridge.pdf')
	plt.close()






def bar_plots_mean():

	number_points_user_train = np.load('../aux_plots/'+name_dataset+'/number_points_user_train.npy') 
	#dict = Counter(number_points_user_train)
	q1, q2 =  np.percentile(number_points_user_train, 10),  np.percentile(number_points_user_train, 20)
	print q1,q2

	dict_baseline_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_baseline.txt')
	dict_baseline = {}

	dict_EB_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_theoretical_EB.txt')
	dict_EB = {}

	dict_ridge_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_ridge.txt')
	dict_ridge = {}
	
	dict_xgb_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_xgboost.txt')
	#dict_xgb = {}

	#dict_xgb_txt = open('../aux_plots/'+name_dataset+'/basic/dict_xgboost.txt')
	dict_xgb = {}

	dict_EB_xgboost_txt = open('../aux_plots/'+name_dataset+'/advanced/dict_alternative_xgboost.txt')
	dict_EB_xgboost = {}

	for (dict_txt, dict_)  in zip([dict_baseline_txt, dict_EB_txt, dict_ridge_txt, dict_xgb_txt, dict_EB_xgboost_txt], [dict_baseline, dict_EB, dict_ridge, dict_xgb, dict_EB_xgboost]):
		for dict_txt in dict_txt:
			dict_txt = dict_txt[1:-1].split(',')

			for points in dict_txt:
				points = points.split(':')
				n_user, MAE   = int(points[0]), float(points[1])
				dict_[n_user] = MAE


	SMALL = q1
	LARGE = q2

	all_baseline_MAE = [0, 0, 0]
	all_EB_MAE = [0, 0, 0]
	all_EB_ridge = [0, 0, 0]
	all_EB_xgb = [0, 0, 0]
	all_EB_xgboost_MAE = [0, 0, 0]

	for (all_MAE, dict_) in zip([all_baseline_MAE, all_EB_MAE, all_EB_ridge, all_EB_xgb, all_EB_xgboost_MAE], [dict_baseline, dict_EB, dict_ridge, dict_xgb, dict_EB_xgboost]):
		for n_train in dict_baseline.keys():
			if n_train <= SMALL:
				all_MAE[0] += dict_[n_train]

			if n_train <= LARGE:
				all_MAE[1] += dict_[n_train]
			#elif n_train <= LARGE_BIS:

			if n_train > LARGE-1:
				all_MAE[2] += dict_[n_train]
			#else:
			#	all_MAE[3] += dict_[n_train]


	copy_baseline = np.copy(all_baseline_MAE)
	for all_MAE in [all_baseline_MAE, all_EB_MAE, all_EB_ridge, all_EB_xgb, all_EB_xgboost_MAE]:
		for i in range(len(all_MAE)):
			all_MAE[i] /= copy_baseline[i]

	print np.round(all_baseline_MAE,3), np.round(all_EB_MAE,3), np.round(all_EB_ridge,3), np.round(all_EB_xgb,3),  np.round(all_EB_xgboost_MAE,3)


	bars_test_errors(all_baseline_MAE, all_EB_MAE, all_EB_xgb, all_EB_xgboost_MAE)
	plt.savefig('../aux_plots/'+name_dataset+'/advanced/bar_plots.pdf',bbox_inches='tight')
	plt.close()



bar_plots_mean()
#features_importance()


