import numpy as np
import pandas as pd

from scipy.stats import norm
import math




########################################## CLASSICAL EM ##########################################

def prob(x, mu, sig, proba):
    return proba*norm.pdf(x, mu, sig)



### E-Step

def expectation(df, parameters):
    all_p_gaussian_1 = []
    for i in range(df.shape[0]):
        p_gaussian_1  = prob(df['y'][i], parameters['mu_1'], parameters['sigma_1'],    parameters['proba_gaussian_1'] )
        p_gaussian_2  = prob(df['y'][i], parameters['mu_2'], parameters['sigma_2'], 1- parameters['proba_gaussian_1'] )    
        p_gaussian_1 /= p_gaussian_1 + p_gaussian_2
        all_p_gaussian_1.append(p_gaussian_1)
    
    df['proba_1'] = all_p_gaussian_1
    return df



### M-Step

def maximization(df, parameters):
    N_total = df.shape[0]
    sum_proba_1, sum_proba_2 = np.sum(df['proba_1']), N_total - np.sum(df['proba_1'])
    parameters['proba_gaussian_1']    = np.sum(df['proba_1']) / N_total

    parameters['mu_1']     = np.sum( df.apply(lambda x: x['y'] * x['proba_1'], axis=1) )
    parameters['mu_1']    /= sum_proba_1
    
    parameters['mu_2']     = np.sum( df.apply(lambda x: x['y'] * (1-x['proba_1']), axis=1) )
    parameters['mu_2']    /= sum_proba_2
   
    parameters['sigma_1']  = np.sum( df.apply(lambda x: (x['y'] - parameters['mu_1'])**2 * x['proba_1'], axis=1) )
    parameters['sigma_1'] /= sum_proba_1
    parameters['sigma_1']  = math.sqrt(parameters['sigma_1'])
    
    parameters['sigma_2']  = np.sum( df.apply(lambda x: (x['y'] - parameters['mu_2'])**2 * (1-x['proba_1']), axis=1) )
    parameters['sigma_2'] /= sum_proba_2
    parameters['sigma_2']  = math.sqrt(parameters['sigma_2'])
    
    return parameters



### Criterion: distance between means

def distance(old_params, new_params):
    dist = np.linalg.norm( [old_params[param] - new_params[param] for param in ['mu_1', 'mu_2']] )
    return dist 



### EM algorithm

def run_EM(data, init_params):

# init_params: first guess of parameters

    df     = pd.DataFrame(data, columns=['y'])
    params = init_params
    
    test_cv = 1e6
    epsilon = 0.01
    iters   = 0
    while test_cv > epsilon:
        iters += 1

        # E-step
        df = expectation(df, params)

        # M-step
        old_params = params.copy()
        params     = maximization(df, params)

        test_cv = distance(params, old_params)
        print("Iteration {}, shift {}".format(iters, test_cv))    
    return params








