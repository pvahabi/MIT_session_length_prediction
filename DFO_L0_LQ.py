import numpy as np

import numpy as np
from   sklearn.linear_model import LinearRegression
from   sklearn.linear_model import Lasso 
from   sklearn.linear_model import Ridge 

import math



# Support functions for the Discrete First Order (DFO) algorithm


############## HIGHEST EIGENVALUE OF XTX ################

# Classical power method for fast computation of the highest eigenvalue of XTX with p large
# Beta is randomly initialized. Then at every iteration, it is multiplied by XTX and normalized

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





############## SOFT THRESHOLDING OPERATORS ################

# Solve argmin 0.5 \| \beta - u \|_2^2 + llambda L(\beta) with L \in {l1, l2, l2^2}

def soft_thresholding_l1(u, llambda):
    return np.array([np.sign(u_i)*max(0, abs(u_i)-llambda) for u_i in u])


def soft_thresholding_l2(u, llambda):
    l2_norm = np.linalg.norm(u)
    return max(0, l2_norm-llambda)/(l2_norm+1e-10)*u

    
def soft_thresholding_l2_2(u, llambda):
    return np.array([u_i/(1.+2*llambda) for u_i in u])






############## SOLVE RETRICTED PROBLEM ON SUPPORT ################

def solve_restricted_problem(type_penalization, X, y, llambda, beta):

# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
    # - for 'l1' or 'l2^2', we use Lasso or Ridge solved with scikit learn
    # - for 'l2', we use a soft thresholding gradient descent on the support
#BETA              : current estimator with support of size K
    
    N,P     = X.shape
    support = np.where(beta!=0)[0]


#---Support not empty
    if support.shape[0] > 0:

    #---Data resticted on support
        X_support = np.array([X[:,support[i]] for i in range(len(support))]).T
        

    #---Solve restricted problem

        if type_penalization == 'l2':
        ### FOR L2 CALL DFO WITHOUT SOLVING ON SUPPORT
            beta_support, _ = DFO.DFO('l2', X_support, y, len(support), llambda, beta_start=beta[support], solve_support=False)

        else:
            if llambda == 0:
                estimator = LinearRegression()
            
            elif type_penalization in {'l1', 'l2^2'}:
            ### CAREFULL : coefficients
                dict_estimator = {'l1':  Lasso(alpha=llambda/float(N)),#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
                                  'l2^2': Ridge(alpha=2.*llambda, fit_intercept=False, solver='svd')} #http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
                estimator = dict_estimator[type_penalization]

            estimator.fit(X_support, y)
            beta_support  = np.array(estimator.coef_)
            


    #---Compute loss
        beta[support]     = beta_support
        obj_val           = 0.5*np.linalg.norm(y - np.dot(X_support, beta_support) )**2
        dict_penalization = {'l1': np.sum(np.abs(beta_support)), 'l2': np.linalg.norm(beta_support), 'l2^2': np.linalg.norm(beta_support)**2}
        obj_val          += llambda*dict_penalization[type_penalization]


#---Support empty
    else:
        obj_val = 0.5*np.linalg.norm(y)**2

    return beta, obj_val








# Implements the Discrete First Order (DFO) algorithm for l1, l2 or l2^2 regularizations



def DFO_nlarge(type_penalization, X, y, llambda, XTX=[], XTy=[], mu_max=0, use_L0=False, K=0, beta_start=[], threshold_CV=1e-4, solve_support=False):
    
# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
# USE_L0           : do we want the L0 regularization
# K, LLAMBDA       : sparsity and regularization parameters
# BETA_START       : warm start (or [])
# THRESHOLD_CV     : convergence threshold
# SOLVE_SUPPORT    : false only when solving on support for l2


#---Parameters
    N_ITER_MAX = 1e3 #maximum number of iterations
    N,P        = X.shape              
    obj_val    = 0

    dict_thresholding = {'l1':   soft_thresholding_l1,
                         'l2':   soft_thresholding_l2,
                         'l2^2': soft_thresholding_l2_2}


#---Check values
    if np.array(XTy).shape[0] == 0: XTy = np.dot(X.T, y)
    if np.array(XTX).shape[0] == 0: XTX = np.dot(X.T, X)
    if mu_max == 0: mu_max = power_method(X)    #highest eigenvalue with power method

    if not use_L0: K=P


#---Intialization
    old_beta   = -np.ones(P)
    beta       = beta_start if np.array(beta_start).shape[0]>0 else np.zeros(P)
    t_AGD_old  =1
    t_AGD      =1
    old_eta    = beta
    eta        = beta



    
    
#---We stop the main loop if the CV criterion is satisfied or after a maximum number of iterations
    test = 0
    while np.linalg.norm(beta-old_beta) > threshold_CV and test < N_ITER_MAX: 
        test += 1
        old_beta = np.copy(beta)

    #---Gradient descent
        #grad = beta - 1./mu_max * (np.dot(X.T, np.dot(X, beta)) - XTy)
        grad = beta - 1./mu_max * (np.dot(XTX, beta) - XTy)
        
    
    
    ### L0 THRESHOLDING IF L0
        if use_L0:
            coefs_sorted = np.abs(grad).argsort()[::-1]
            for idx in coefs_sorted[K:]: eta[idx] = 0

            grad_thresholded = dict_thresholding[type_penalization](grad[coefs_sorted[:K]], llambda/mu_max) 
            eta[coefs_sorted[:K]] = grad_thresholded
        
        else:
            eta = dict_thresholding[type_penalization](grad, llambda/mu_max) 



    #---Apply the soft-thresholding operator for each of the K0 highest coefficients
        
    ### AGD IF NOT L0
        if use_L0:
            beta = eta
        else:
            t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
            aux_t_AGD = (t_AGD_old-1)/t_AGD

            beta      = eta + aux_t_AGD * (eta - old_eta)
            t_AGD_old = t_AGD
            old_eta   = eta


#---Solve restricted problem in the support
    print 'Number iter for DFO', test
    if solve_support: beta, obj_val = solve_restricted_problem(type_penalization, X, y, llambda, beta)
    return np.array(beta)


