



######################## TEST


## gaussian

def likelihood_gaussian_test(y_datas, params):

    likelihood = 0
    for y_data in y_datas: likelihood += (y_data - params['m'])**2  /(2*params['sigma']**2) + math.log(params['sigma'])
    return likelihood
    
    return likelihood





## mixture

def expectation_test(y_datas, params):
    probas = []
    for y_data in y_datas:
        p_gaussian_1  = prob(y_data, params['m_1'], params['sigma_1'],    params['proba_gaussian_1'] )
        p_gaussian_2  = prob(y_data, params['m_2'], params['sigma_2'], 1- params['proba_gaussian_1'] )    
        p_gaussian_1 /= p_gaussian_1 + p_gaussian_2
        probas.append(p_gaussian_1)
    return probas


def likelihood_mixture(y_datas, params, probas):
    likelihood = 0
    for j in range(len(y_datas)):
        likelihood +=    probas[j] * ( (y_datas[j] - params['m_1'])**2/(2*params['sigma_1']**2) + math.log(params['sigma_1']) )
        likelihood += (1-probas[j])* ( (y_datas[j] - params['m_2'])**2/(2*params['sigma_2']**2) + math.log(params['sigma_2']) )
    return likelihood


def likelihood_mixture_test(y_datas, params):

    ### Probas
    probas = expectation_test(y_datas, params)
    
    ### Compute residuals
    likelihood  = likelihood_mixture(y_datas, params, probas)
    
    return likelihood









## mixture with priors

def likelihood_prior(y_data_users, means_listeners_1, means_listeners_2, params, probas_users):

    likelihood   = 0
    sig_A, sig_B = 1./(params['lambda_1']*params['sigma_1']), 1./(params['lambda_2']*params['sigma_2'])

    for i in range(len(y_data_users)):
        y_data_user   = y_data_users[i]
        proba_user    = probas_users[i]
        
        mean_listener_1 = means_listeners_1[i]
        mean_listener_2 = means_listeners_2[i]

        likelihood += (mean_listener_1 - params['m_1'])/2*sig_A**2 + math.log(sig_A)
        likelihood += (mean_listener_2 - params['m_2'])/2*sig_B**2 + math.log(sig_B)

        for j in range(len(y_data_user)):
            likelihood +=    proba_user[j] * ( (y_data_user[j] - mean_listener_1)**2/(2*params['sigma_1']**2) + math.log(params['sigma_1']) )
            likelihood += (1-proba_user[j])* ( (y_data_user[j] - mean_listener_2)**2/(2*params['sigma_2']**2) + math.log(params['sigma_2']) )

            likelihood += proba_user[j] * math.log(params['proba_gaussian_1']) + (1-proba_user[j]) * math.log(1-params['proba_gaussian_1'])
    
    return likelihood



def likelihood_prior_test(idx_data, y_data, params):
    df_data      = pd.DataFrame([idx_data, y_data ], index=['listener_id', 'y']).T
    listeners_id = pd.unique(df_data['listener_id'])
    N_total      = df_data.shape[0]
    
    y_data_users = []
    for listener_id in listeners_id:
        df_data_listener = df_data[df_data['listener_id'] == listener_id]
        y_data_users.append(list(df_data_listener['y'].values))
    

    ### Test probas
    probas_users  = expectation_with_prior(y_data_users, 
                                           params['means_probas']['mean_listener_1'].values, 
                                           params['means_probas']['mean_listener_2'].values, 
                                           params)
    print probas_users[0]
    
    ### Likelihood
    likelihood     = likelihood_prior(y_data_users, 
                                    params['means_probas']['mean_listener_1'].values, 
                                    params['means_probas']['mean_listener_2'].values, 
                                    params, 
                                    probas_users)
    return likelihood   
















## mixture with priors and covariates

def likelihood_prior_cov(residuals_users, means_listeners_1, means_listeners_2, params, probas_users, add_reg=True):

    likelihood   = params['alpha_ridge'] * np.linalg.norm(params['beta'])**2 if add_reg else 0
    sig_A, sig_B = 1./(params['lambda_1']*params['sigma_1']), 1./(params['lambda_2']*params['sigma_2'])

    for i in range(len(residuals_users)):
        residual_user   = residuals_users[i]
        proba_user      = probas_users[i]
        
        mean_listener_1 = means_listeners_1[i]
        mean_listener_2 = means_listeners_2[i]

        likelihood += (mean_listener_1 - params['m_1'])/2*sig_A**2 + math.log(sig_A)
        likelihood += (mean_listener_2 - params['m_2'])/2*sig_B**2 + math.log(sig_B)

        for j in range(len(residual_user)):
            likelihood +=    proba_user[j] * ( (residual_user[j] - params['m_1'])**2/(2*params['sigma_1']**2) + math.log(params['sigma_1']) )
            likelihood += (1-proba_user[j])* ( (residual_user[j] - params['m_2'])**2/(2*params['sigma_2']**2) + math.log(params['sigma_2']) )

            likelihood += proba_user[j] * math.log(params['proba_gaussian_1']) + (1-proba_user[j]) * math.log(1-params['proba_gaussian_1'])
    
    return likelihood



def likelihood_prior_cov_test(idx_data, x_data_test, y_data_test, params):
    df_data      = pd.DataFrame([idx_data, x_data_test, y_data_test ], index=['listener_id', 'x', 'y']).T
    listeners_id = pd.unique(df_data['listener_id'])
    N_total      = df_data.shape[0]
    
    x_data_users = []
    y_data_users = []
    for listener_id in params['means']['listener_id'].values:
        df_data_listener = df_data[df_data['listener_id'] == listener_id]
        x_data_users.append(list(df_data_listener['x'].values)) ### [] if empty
        y_data_users.append(list(df_data_listener['y'].values)) ### [] if empty
    

    ### Test residuals
    residuals_users   = [[y_data_users[i][j] - np.dot(x_data_users[i][j], params['beta']) for j in range(len(y_data_users[i]))] for i in range(len(y_data_users))]
    
    ### Test probas
    probas_users      = expectation_with_prior(y_data_users, 
                                               params['means']['mean_listener_1'].values, 
                                               params['means']['mean_listener_2'].values, 
                                               params)
    
    ### Likelihood
    likelihood = likelihood_prior_cov(residuals_users, 
                                    params['means']['mean_listener_1'].values, 
                                    params['means']['mean_listener_2'].values, 
                                    params, 
                                    probas_users, 
                                    add_reg=True)
    return likelihood

