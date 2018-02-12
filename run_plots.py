import numpy as np
import matplotlib.pyplot as plt

import plt



def run_plots():

######## RIDGE ########

	plot_column_importance(x_train_columns, dict_best_params_ridge['beta'])
	plt.savefig('../results/'+name_dataset+'/'+is_basic+'/coefficients_importance_'+dict_word1[theoretical_alpha]+'.pdf', bbox_inches='tight')
	plt.close()

	#plot_errors_by_sessions(dict_best_params_ridge['sessions_errors'])
	#plt.savefig('../results/errors_by_sessions_ridge.pdf', bbox_inches='tight')
	#plt.close()


######## ALT MIN RIDGE ########

	plot_column_importance(x_train_columns, dict_best_params_alt_min_ridge['beta'])
	plt.savefig('../results/'+name_dataset+'/'+is_basic+'/coefficients_importance_alternative_ridge.pdf', bbox_inches='tight')
	plt.close()



######## ALT MIN LASSO ########

	plot_column_importance(x_train_columns, dict_best_params_alt_min_lasso['beta'])
	plt.savefig('../results/'+name_dataset+'/'+is_basic+'/coefficients_importance_alternative_lasso.pdf', bbox_inches='tight')
	plt.close()



######## PLOT ERRORS BY SESSIONS ########	
	
	plot_errors_by_sessions(dict_EB_pairs_sessions_errors, dict_best_params_ridge, dict_best_params_alt_min_ridge)
	plt.savefig('../results/'+name_dataset+'/'+is_basic+'/errors_by_number_sessions.pdf', bbox_inches='tight')
	plt.close()

	all_names = ['EB', 'ridge', 'EB-ridge']
	all_dicts = [dict_EB_pairs_sessions_errors, dict_best_params_ridge, dict_best_params_alt_min_ridge]
	for i in range(3):
		g = open('../results/'+name_dataset+'/'+is_basic+'/'+all_names[i]+'.txt',"w")
		g.write( str(all_dicts[i]) )
		g.close()