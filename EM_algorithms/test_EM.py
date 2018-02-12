import numpy as np
import pandas as pd
import random as rand

from scipy.stats import norm
import math


from EM_classical import *
from EM_with_prior import *
from EM_with_prior_and_cov import *





def test_EM_with_prior():

######### SIMULATE TRUTH #########
	users_id    = ['aa','bb','cc','dd']
	true_values = {'m_1':3, 'm_2':8, 's_1':.5, 's_2':.5, 'sigma_1':.2, 'sigma_2':1, 'proba_gaussian_1':.3}

	N_USERS    = len(users_id)
	N_BY_USERS = 2000


### True parameters for each user
	means_gaussian_1 = np.random.normal(true_values['m_1'], true_values['s_1'], N_USERS)
	means_gaussian_2 = np.random.normal(true_values['m_2'], true_values['s_2'], N_USERS)


### Simulate same number of points in each class for each user
	idx_data, y_data = [], []
	for idx in range(N_USERS):
		y1 = np.random.normal(means_gaussian_1[idx], true_values['sigma_1'], int(N_BY_USERS*   true_values['proba_gaussian_1']))
		y2 = np.random.normal(means_gaussian_2[idx], true_values['sigma_2'], int(N_BY_USERS*(1-true_values['proba_gaussian_1'])))
		                      
		idx_data += [users_id[idx]]*N_BY_USERS
		y_data   += list(y1) + list(y2)
	                          



######### RUN EM WITH PRIOR #########
	init_params = {'m_1':2.5, 'm_2':7.5, 'lambda_1':.3, 'lambda_2':1.5, 'sigma_1':.2, 'sigma_2':1.2, 'proba_gaussian_1':.2}

	params = process_and_run_EM_with_prior(np.array(idx_data), np.array(y_data), np.array(y_data), init_params, 1e-2)
	N_BY_SAMPLES = 20000 # 10 times number of points
	data_mixture = []


### Simulate points according to estimated parameters
	for idx in range(N_USERS):
		mixture_1     = np.random.normal(params['means_probas']['mean_listener_1'][idx], params['sigma_1'], int(N_BY_USERS*   params['means_probas']['proba_gaussian_1'][idx]))
		mixture_2     = np.random.normal(params['means_probas']['mean_listener_2'][idx], params['sigma_2'], int(N_BY_USERS*(1-params['means_probas']['proba_gaussian_1'][idx])))
		data_mixture += list(mixture_1) + list(mixture_2)


### Plot and compare distributions
	fig = plt.figure(figsize=(15,5))
	n, bins, patches = plt.hist(y_data, 70, normed=True)
	n, bins, patches = plt.hist(data_mixture, 70, normed=True, histtype=u'step', linewidth=2)
	plt.savefig('../results/EM_with_prior_synthetic.pdf', bbox_inches='tight')
	plt.close()








def test_EM_with_prior_and_cov():

######### SIMULATE TRUTH #########
	users_id    = ['aa','bb','cc','dd']
	true_values = {'m_1':3, 'm_2':8, 's_1':.5, 's_2':.5, 'sigma_1':.2, 'sigma_2':1, 'proba_gaussian_1':.3}

	N_USERS    = len(users_id)
	N_BY_USERS = 2000
	N_COVS     = 2


### True means for each user
	means_gaussian_1 = np.random.normal(true_values['m_1'], true_values['s_1'], N_USERS)
	means_gaussian_2 = np.random.normal(true_values['m_2'], true_values['s_2'], N_USERS)
	true_probas      = []


### True covariates and estimators
	X = np.random.binomial(1, .3, (N_BY_USERS*N_USERS, N_COVS))
	beta   = [1, 3]
	X_beta = np.dot(X, beta)


### Simulate same number of points in each class for each user
	idx_data, y_data, yy_data = [], [], []

	for idx in range(N_USERS):
	    X_beta_user  = X_beta[idx*N_BY_USERS : (idx+1)*N_BY_USERS]
	    n_gaussian_1 = int(N_BY_USERS*   true_values['proba_gaussian_1'])
	    
	    y1 = X_beta_user[:n_gaussian_1] + np.random.normal(means_gaussian_1[idx], true_values['sigma_1'], n_gaussian_1)
	    y2 = X_beta_user[n_gaussian_1:] + np.random.normal(means_gaussian_2[idx], true_values['sigma_2'], N_BY_USERS -n_gaussian_1)
	                          
	    idx_data += [users_id[idx]]*N_BY_USERS
	    y_data   += list(y1) + list(y2)
	    
	    ### Help
	    #yy_data     += list(y1-X_beta_user[:n_gaussian_1]) + list(y2-X_beta_user[n_gaussian_1:])
	    #true_probas.append( [1 for _ in range(n_gaussian_1)] + [0 for _ in range(N_BY_USERS -n_gaussian_1)] )



######### RUN EM WITH PRIOR #########
	x_data      = list(X)
	init_params = {'m_1':2.5, 'm_2':7.5, 'lambda_1':.3, 'lambda_2':1.5, 'sigma_1':.2, 'sigma_2':1.2, 'proba_gaussian_1':.2, 'alpha_ridge':5*1e1}
	params = run_EM_with_prior_and_cov(idx_data, x_data, y_data, init_params, 1e-2)
	print params['beta'],  params['means'], params['sigma_1'], params['sigma_2']


### Simulate points according to estimated parameters
	N_BY_SAMPLES = 200
	data_mixture = []
	    
	for idx_user in range(N_USERS):
	    for j in range(len(params['proba_G1_session'][idx])):
	        number_G1_session = int(N_BY_SAMPLES * params['proba_G1_session'][idx_user][j])
	        X_beta_session    = np.dot(params['x_user_session'][idx_user][j], params['beta'])
	        
	        X_beta_arr     = np.array([ np.array(X_beta_session) for _ in range(number_G1_session) ])
	        mixture_1      = X_beta_arr + np.random.normal(params['means']['mean_listener_1'][idx_user], params['sigma_1'], number_G1_session)
	        
	        X_beta_arr     = np.array([ np.array(X_beta_session) for _ in range(N_BY_SAMPLES - number_G1_session) ])
	        mixture_2      = X_beta_arr + np.random.normal(params['means']['mean_listener_2'][idx_user], params['sigma_2'], N_BY_SAMPLES - number_G1_session)
	        
	        data_mixture  += list(mixture_1) + list(mixture_2)


### Plot and compare distributions
	fig = plt.figure(figsize=(15,5))
	n, bins, patches = plt.hist(y_data, 70, normed=True)
	n, bins, patches = plt.hist(data_mixture, 70, normed=True, histtype=u'step', linewidth=2)
	plt.savefig('../results/EM_with_prior_and_cov_synthetic.pdf', bbox_inches='tight')
	plt.close()




#test_EM_with_prior()
#test_EM_with_prior_and_cov()






