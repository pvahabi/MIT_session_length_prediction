import numpy as np
import pandas as pd

from scipy.stats import norm
import math

from EM_classical import *



########################################## E AND M STEPS ##########################################


### E-step with prior

def expectation_with_prior(y_data_users, means_listeners_1, means_listeners_2, params):

# probas_users: contains proba of session j from user i to belong to first mixture

	probas_users = [[] for i in range(len(y_data_users))]
	
	for i in range(len(y_data_users)):
		y_data_user     = y_data_users[i]
		mean_listener_1 = means_listeners_1[i]
		mean_listener_2 = means_listeners_2[i]
		
		for y_data in y_data_user:
			p_gaussian_1  = prob(y_data, mean_listener_1, params['sigma_1'],    params['proba_gaussian_1'])
			p_gaussian_2  = prob(y_data, mean_listener_2, params['sigma_2'], 1- params['proba_gaussian_1'] )    
			p_gaussian_1 /= p_gaussian_1 + p_gaussian_2
			
			probas_users[i].append(p_gaussian_1)
	return probas_users




### M-step with prior
	
def maximization_with_prior(y_data_users, probas_users, means_listeners_1, means_listeners_2, params, N_total):

	sum_probas = np.sum([np.sum(proba_user) for proba_user in probas_users])
	sum_proba_1, sum_proba_2 = sum_probas, N_total - sum_probas
	new_sigma_1, new_sigma_2 = 0,0
	
	for i in range(len(y_data_users)):
		y_data_user  = y_data_users[i]
		proba_user   = probas_users[i]
		
		sum_proba_1_listener, sum_proba_2_listener = np.sum(proba_user), len(proba_user) - np.sum(proba_user)
		
		means_listeners_1[i]  = np.sum(np.array(y_data_user) * np.array(proba_user)) + params['m_1'] * params['lambda_1']**2 
		means_listeners_1[i] /= sum_proba_1_listener + params['lambda_1']**2 

		means_listeners_2[i]  = np.sum(np.array(y_data_user) * (np.ones(len(y_data_user)) - np.array(proba_user))) + params['m_2'] * params['lambda_2']**2 
		means_listeners_2[i] /= sum_proba_2_listener + params['lambda_2']**2
		
		new_sigma_1 += np.sum( [ (y_data_user[j] - means_listeners_1[i])**2 *    proba_user[j]  for j in range(len(y_data_user)) ])
		new_sigma_2 += np.sum( [ (y_data_user[j] - means_listeners_2[i])**2 * (1-proba_user[j]) for j in range(len(y_data_user)) ])
		   
	new_sigma_1 /= sum_proba_1
	new_sigma_2 /= sum_proba_2
	
	params['proba_gaussian_1'] = sum_proba_1 / N_total
	params['sigma_1']          = math.sqrt(new_sigma_1)
	params['sigma_2']          = math.sqrt(new_sigma_2)
	
	return means_listeners_1, means_listeners_2, params




### Criterion: distance between means

def distance_with_prior(means_listeners_1, means_listeners_2, old_means_listeners_1, old_means_listeners_2):
	dist  = np.linalg.norm(means_listeners_1 - old_means_listeners_1) + np.linalg.norm(means_listeners_2 - old_means_listeners_2)
	dist /= means_listeners_1.shape[0]
	return dist 







########################################## EM WITH PRIOR ON THE PARAMETERS ##########################################


def run_EM_with_prior(y_data_users, means_listeners_1, means_listeners_2, params, N_total, epsilon):

# INPUT:  - the means m_1 & m_2 of the priors, lambda_1 & lambda_2 the ratios sigma_1^2/A^2 & sigma_2^2/B^2
#         - all the sessions (and associated users)

# OUTPUT: the means, proba_gaussian_1, sigma_1, sigma_2

	test_cv = 1e6
	iters   = 0
	while test_cv > epsilon:
		iters += 1

	### E-step
		probas_users = expectation_with_prior(y_data_users, means_listeners_1, means_listeners_2, params)

	### M-step
		old_means_listeners_1, old_means_listeners_2 = list(means_listeners_1), list(means_listeners_2) 
		means_listeners_1, means_listeners_2, params = maximization_with_prior(y_data_users, 
																			   probas_users, 
																			   means_listeners_1, 
																			   means_listeners_2, 
																			   params, 
																			   N_total)

		test_cv = distance_with_prior(means_listeners_1, means_listeners_2, old_means_listeners_1, old_means_listeners_2)
		print "EM iteration {}, test cv {}".format(iters, test_cv)
   
	### Return updated probas
	probas_users = expectation_with_prior(y_data_users, means_listeners_1, means_listeners_2, params)
	return params, probas_users, means_listeners_1, means_listeners_2








########################################## PROCESS AND RUN ##########################################

# We process the data from the original DF so that we can run the EM above

def process_and_run_EM_with_prior(idx_data, x_data, y_data, init_params, epsilon):

	_, y_data_users, listeners_id = process(idx_data, x_data, y_data)
	N_total = len(y_data)

### Init all the parameters
	means_listeners_1 = init_params['m_1']*np.ones(len(listeners_id))
	means_listeners_2 = init_params['m_2']*np.ones(len(listeners_id))
	params            = init_params
	


	params, probas_users, means_listeners_1, means_listeners_2 = run_EM_with_prior(y_data_users, 
																				   means_listeners_1, 
																				   means_listeners_2, 
																				   params, 
																				   N_total, 
																				   epsilon)


	sum_probas_users = np.array([np.mean(probas_user) for probas_user in probas_users])
	params['means_probas'] = pd.DataFrame([listeners_id, means_listeners_1, means_listeners_2, sum_probas_users], index=['listener_id', 'mean_listener_1', 'mean_listener_2', 'proba_gaussian_1']).T
	return params






def process(idx_data, x_data, y_data):

### Split data into users -> this will speed up the EM
	df_data      = pd.DataFrame([idx_data, x_data, y_data, [0 for _ in range(len(y_data)) ] ], index=['listener_id', 'x', 'y', 'proba_1']).T
	listeners_id = pd.unique(df_data['listener_id'])
	N_total      = df_data.shape[0]
	
	x_data_users = []
	y_data_users = []
	for listener_id in listeners_id:
		df_data_listener = df_data[df_data['listener_id'] == listener_id]
		x_data_users.append(list(df_data_listener['x'].values))
		y_data_users.append(list(df_data_listener['y'].values))
	return x_data_users, y_data_users, listeners_id




