import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from sklearn.linear_model import Ridge

from EM_with_prior import *





########################################## RIDGE ON RESIDUALS ##########################################

def ridge_adapted_covariates(x_data_users, y_data_users, means_listeners_1, means_listeners_2, params, probas_users):

# INPUT:  means_listeners_1, means_listeners_2, params, probas_users obtained from EM
# OUTPUT: beta_train, the Ridge estimator on the residuals
    

### Define X_tilde, y_tilde
    x_tilde = []
    y_tilde = []
    
    for i in range(len(y_data_users)):
        x_data_user     = x_data_users[i]
        y_data_user     = y_data_users[i]
        proba_user      = probas_users[i]
        mean_listener_1 = means_listeners_1[i]
        mean_listener_2 = means_listeners_2[i]
        
        for j in range(len(proba_user)):
            weight_session_j  = math.sqrt(proba_user[j]/params['sigma_1']**2 + (1-proba_user[j])/params['sigma_2']**2)  
            x_data_user_tilde = weight_session_j* x_data_user[j]                    
            y_data_user_tilde = (1./weight_session_j)* (proba_user[j]/params['sigma_1']**2 * (y_data_user[j] - mean_listener_1) + (1-proba_user[j])/params['sigma_2']**2 * (y_data_user[j] - mean_listener_2)) 
            
            x_tilde.append(x_data_user_tilde)
            y_tilde.append(y_data_user_tilde)
        

### Run the Ridge
    estimator = Ridge(alpha=params['alpha_ridge'], fit_intercept=False, solver='svd')
    estimator.fit(x_tilde, y_tilde)
    beta_train = np.copy(estimator.coef_)
    return beta_train   






########################################## TRAIN LOG-LIKELIHOOOD ##########################################

def compute_obj_val(residuals_users, means_listeners_1, means_listeners_2, params, probas_users, beta_train):

    obj_val_unit = params['alpha_ridge'] * np.linalg.norm(beta_train)**2
    
    sig_A, sig_B = 1./(params['lambda_1']*params['sigma_1']), 1./(params['lambda_2']*params['sigma_2'])

    for i in range(len(residuals_users)):
        residual_user   = residuals_users[i]
        proba_user      = probas_users[i]
        
        mean_listener_1 = means_listeners_1[i]
        mean_listener_2 = means_listeners_2[i]

        obj_val_unit += (mean_listener_1 - params['m_1'])/2*sig_A**2 + math.log(sig_A)
        obj_val_unit += (mean_listener_2 - params['m_2'])/2*sig_B**2 + math.log(sig_B)

        for j in range(len(residual_user)):
            obj_val_unit +=    proba_user[j] /(2*params['sigma_1']**2) * (residual_user[j] - mean_listener_1)**2 
            obj_val_unit += (1-proba_user[j])/(2*params['sigma_2']**2) * (residual_user[j] - mean_listener_2)**2 

            obj_val_unit += proba_user[j] * math.log(params['proba_gaussian_1']) + (1-proba_user[j]) * math.log(1-params['proba_gaussian_1'])
    
    return obj_val_unit







########################################## PROCESS AND RUN ALTERNATIVE MINIMIZATION ##########################################


def run_EM_with_prior_and_cov(idx_data, x_data, y_data, init_params, epsilon):
    
### Process data
    x_data_users, y_data_users, listeners_id = process(idx_data, x_data, y_data)
    N_total = len(y_data)
    

### Init all the parameters
    means_listeners_1 = init_params['m_1']*np.ones(len(listeners_id))
    means_listeners_2 = init_params['m_2']*np.ones(len(listeners_id))
    params            = init_params
    
    beta_train        = np.zeros(x_data[0].shape)
    residuals_users   = [list(y_data_user) for y_data_user in y_data_users]
    

### Main loop
    old_obj_val   =  1e10
    obj_val       = -1e10
    ratio_obj_val =  1e10
    loop          = 0
    
    while loop < 10 and ratio_obj_val > 1e-2:
        loop += 1
        
    ### Minimize for parameters with fixed beta
        params, probas_users, means_listeners_1, means_listeners_2 = run_EM_with_prior(residuals_users, 
                                                                                       means_listeners_1, 
                                                                                       means_listeners_2, 
                                                                                       params,
                                                                                       N_total,
                                                                                       epsilon)
    ### Minimize for beta with fixed parameters
        beta_train = ridge_adapted_covariates(x_data_users, 
                                              y_data_users, 
                                              means_listeners_1, 
                                              means_listeners_2, 
                                              params, 
                                              probas_users)

    ### Compute residuals
        residuals_users = [[y_data_users[i][j] - np.dot(x_data_users[i][j], beta_train) for j in range(len(y_data_users[i]))] for i in range(len(y_data_users))]


    ### Compute train likelihood
        old_obj_val   = obj_val
        obj_val       = compute_obj_val(residuals_users, means_listeners_1, means_listeners_2, params, probas_users, beta_train)
        ratio_obj_val = abs( (old_obj_val- obj_val) / float(old_obj_val) )
        print "AM main loop {}, test cv {}\n".format(loop, ratio_obj_val)

    

### Get probas
    params, probas_users, means_listeners_1, means_listeners_2 = run_EM_with_prior(residuals_users, means_listeners_1, means_listeners_2, params, N_total, epsilon)
    params['means'] = pd.DataFrame([listeners_id, means_listeners_1, means_listeners_2], index=['listener_id', 'mean_listener_1', 'mean_listener_2']).T
    params['beta']  = beta_train
    
    params['proba_G1_session'] = probas_users
    params['x_user_session']   = x_data_users
    return params











    