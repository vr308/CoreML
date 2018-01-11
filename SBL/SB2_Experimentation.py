#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:53:52 2017

@author: vidhi.lalchand
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SB2_Likelihoods import SB2_Likelihoods
from SB2_UserOptions import SB2_UserOptions
from SB2_ParameterSettings import SB2_ParameterSettings
from SparseBayes import SparseBayes
from sklearn.preprocessing import normalize

#######################################################################
#
# Support function to compute basis
#
#######################################################################

def univariate_spline_kernel(x_m,x_n):
    
    min_x = np.minimum(x_m,x_n)
    return 1 + x_m*x_n + x_m*x_n*min_x - (x_m+x_n)*np.power(min_x,2)/2.0 + np.power(min_x,3)/3.0
    
def polynomialBasis(X):
    Xa = np.asarray(X)
    X1 = np.asarray(np.ones(shape=(np.size(Xa),1)))
    Xa2 = Xa*Xa
    Xa3 = Xa2*Xa
    Xa4 = Xa3*Xa
    Xa5 = Xa3*Xa2
    Xa6 = Xa5*Xa
    Xa7 = Xa5*Xa2
    return(np.matrix(np.column_stack((X1,Xa,Xa2,Xa3,Xa4,Xa5,Xa6,Xa7))))


def distSquared(X,Y):
    nx = np.size(X, 0)
    ny = np.size(Y, 0)
    D2 =  ( np.multiply(X, X).sum(1)  * np.ones((1, ny)) ) + ( np.ones((nx, 1)) * np.multiply(Y, Y).sum(1).T  ) - 2*X*Y.T
    return D2

def generate_gaussian_basis(X,Y, bw): 
        return np.matrix(np.exp(-distSquared(X,Y)/(bw**2)))
    

#def generate_posterior_w_std(alphas, beta, BASIS):
    
#    full_alphas = np.zeros(100)
#    full_alphas.fill(np.infty)
#    for i in alphas.index:
#        full_alphas[i-1] = alphas[i]    
#    
#    A = np.zeros((M,M))
#    np.fill_diagonal(A,full_alphas)
#    Sigma_inv = np.linalg.inv(A + beta*np.dot(BASIS.T, BASIS))

#     inv_var = []
#     for i in alphas.index:
#         inv_var.append(alphas[i] + beta*np.dot(BASIS[:,i-1].T,BASIS[:,i-1]))
#     w_var = np.divide(1,inv_var)
#     return np.sqrt(w_var)  
        

if __name__ == "__main__":
    
    rseed = 87
    np.random.seed(rseed)
    
    N = 100
    noiseToSignal = 0.4
    
    # Uniformly spaced or randomly spaced input data 
    #X = np.matrix(np.random.uniform(-np.pi,np.pi,N)).T
    
    # Sin(x)x results
    #X = np.matrix(np.sort(np.random.uniform(-10,10,N))).T # random
    #X = np.matrix(np.linspace(-10,10,N)).T #uniform
    #z = np.sin(X)/X
    
    # Composite function result
    
    u = np.sort(np.random.uniform(-2,4,N))
    #u = np.linspace(1,20,100)
    X = np.matrix(u).T # Sampling at random locations
    #z = np.matrix(0.5*np.sin(u) + 0.5*u -0.02*(u-5)**2).T
    z = np.multiply(np.sin(np.power(X,2)),X)
    noise = np.std(z, ddof=1) * noiseToSignal
    Outputs = z + noise*np.random.randn(N,1)

    
 ##################################################################
 #  Basis Generation 
 ##################################################################
 
    bw = 2
    BASIS = np.matrix(np.exp(-distSquared(X,X)/(bw**2)))
    
    M = BASIS.shape[1]
    likelihood_ = 'Gaussian'
    
    #BASIS = np.matrix(normalize(BASIS, norm='l2', axis=0))
     
 #######################################################################
    # 
    # --- SPARSE BAYES INFERENCE SECTION ---
    # 
    #######################################################################
    #
    # The section of code below is the main section required to run the
    # SPARSEBAYES algorithm.
    # 
    #######################################################################
    #
    # Set up the options:
    # 
    # - we set the diagnostics level to 2 (reasonable)
    # - we will monitor the progress every 10 iterations
    # 
    iterations = 500
    
    OPTIONS = SB2_UserOptions('ITERATIONS', iterations, 'DIAGNOSTICLEVEL', 2, 'MONITOR', 10)
    
    # Set initial parameter values:
    # 
    # - this specification of the initial noise standard deviation is not
    # necessary, but included here for illustration. If omitted, SPARSEBAYES
    # will call SB2_PARAMETERSETTINGS itself to obtain an appropriate default
    # for the noise (and other SETTINGS fields). NOTE. 
    
    SETTINGS    = SB2_ParameterSettings('NOISESTD', 0.1)
    
    # Now run the main SPARSEBAYES function
        
    B = np.matrix(BASIS)

    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = SparseBayes(likelihood_, BASIS, Outputs, OPTIONS, SETTINGS)

    BASIS = B

    print('\nPARAMETER = \n')
    for key, value in PARAMETER.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))


    print('\nHYPERPARAMETER = \n')
    for key, value in HYPERPARAMETER.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))

    print('\nDIAGNOSTIC = \n')
    for key, value in DIAGNOSTIC.items():
        try:
            print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0,0])))
        except IndexError:
            try:
                print('\t{0} : [{1} {2}]'.format(key,value.shape, type(value[0])))
            except IndexError:
                print('\t{0} : {1}'.format(key,value))
        except AttributeError:
            print('\t{0} : {1}'.format(key,value))
    print('\n')
    
    # Manipulate the returned weights for convenience later
    
    w_infer                             = np.zeros((M,1))
    w_infer[PARAMETER['RELEVANT']]      = PARAMETER['VALUE']
    
    rn = len(PARAMETER['RELEVANT'])
    alphas = pd.Series(index=PARAMETER['RELEVANT'],data=np.array(HYPERPARAMETER['ALPHA']).reshape(rn,))
    beta = HYPERPARAMETER['BETA']
    
    #w_var = np.divide(1,HYPERPARAMETER['ALPHA'])
    #w_std = generate_posterior_w_std(alphas, beta, BASIS)
    w_std = np.array(np.sqrt(PARAMETER['VARIANCE'].diagonal())).reshape(rn,)
    
    # Compute the inferred prediction function
    
    y = BASIS*w_infer
    
    # Test data 
    
    t = np.sort(np.random.uniform(1,20,40))
    #u = np.linspace(1,20,100)
    Xt = np.matrix(t).T # Sampling at random locations
    zt = np.matrix(0.5*np.sin(t) + 0.5*t -0.02*(t-5)**2).T
    noise_t = np.std(z, ddof=1) * noiseToSignal
    Outputs_t = zt + noise_t*np.random.randn(40,1)
   
    # Get the relevant training points and compute BASIS_test with those basis functions
    X_relevant = X[PARAMETER['RELEVANT']]
    W_relevant =  w_infer[PARAMETER['RELEVANT']] 
    
    BASIS_test = np.matrix(np.exp(-distSquared(Xt, X_relevant)/(bw**2)))
    
    # Inferred test prediction
    
    y_test = BASIS_test*W_relevant
    
    # Compute variance around the predictions
    
    pred_var = []
    pred_var_test = []
    rel_index = [x-1 for x in alphas.index]
    rel_phi = BASIS[:,rel_index]
    rel_phi_test = BASIS_test
    for i in np.arange(0,100):
        pred_var.append(np.float(1.0/beta + np.dot(np.dot(rel_phi[i,:],PARAMETER['VARIANCE']),rel_phi[i,:].T)))
    for i in np.arange(0,40):
        pred_var_test.append(np.float(1.0/beta + np.dot(np.dot(rel_phi_test[i,:],PARAMETER['VARIANCE']),rel_phi_test[i,:].T)))

    pred_std = np.matrix(np.sqrt(pred_var).reshape(100,1))
    pred_std_test = np.matrix(np.sqrt(pred_var_test).reshape(40,1))
    
    # Collect and print test results
    
    test_error = np.round(np.sum(np.square(zt - y_test)),2)
    likelihood = DIAGNOSTIC['LIKELIHOOD'][-1]
    
    # Plotting test results
    
    #Extend the curve to plot to cover area not covered in training
    
#    plt.figure()
#    plt.plot(X,z, color='b')
#    #plt.plot(ex,exf, color='b')
#    plt.plot(Xt,Outputs_t,'k*',markersize=4)
#    plt.plot(Xt, y_test, 'r')
#
#    Xt_array = np.array(Xt.reshape(40,))[0]
#    upt_array = np.array((y_test - pred_std_test).reshape(40,))[0]
#    downt_array = np.array((y_test + pred_std_test).reshape(40,))[0]
#    plt.fill_between(Xt_array, upt_array, downt_array, color='red',alpha=0.4, label = '1 sd')
#    plt.plot(X,BASIS[:,PARAMETER['RELEVANT']], color='m')
#    plt.scatter(list(X[PARAMETER['RELEVANT']]),list(Outputs[PARAMETER['RELEVANT']]), c = 'g', marker='+', label='Relevant points')
#    plt.title('Predictions on test data' + ' (Basis width = ' + str(bw) + ')', fontsize='small')
#    plt.legend(fontsize='small')
#    plt.annotate('Test error = ' + str(test_error), xy=(1.0,5.5),fontsize='small')
    

    # Sparse Bayesian Learning Results

    # Plot the generative function and noisy output 
    
    plt.figure(figsize=(10,8))
    plt.subplot(321)
    plt.plot(X,z, color='b')
    plt.ylabel('f(X)')
    plt.plot(X,Outputs,'ko',markersize=2)
    plt.title('Generative function/Noisy Outputs',fontsize='small')
    
    # Plot the Generative function and the predictions
    
    plt.subplot(322)
    plt.plot(X,z,label='Generative function',linewidth=3, color='b')
    plt.plot(X,y, label='Predictive linear model',color='r')
    plt.title('Generative and predictive linear model',fontsize='small')
    plt.legend(fontsize='small',loc=4)
    
    # Plot the relevant vectors and the predictions w error bars 
    
    plt.subplot(323)
    X_array = np.array(X.reshape(100,))[0]
    up_array = np.array((y-pred_std).reshape(100,))[0]
    down_array = np.array((y + pred_std).reshape(100,))[0]
    plt.plot(X,y,label='Predictive linear model',color='r')
    plt.fill_between(X_array, up_array, down_array, color='orange',alpha=0.4, label = '1 sd')
    plt.scatter(list(X[PARAMETER['RELEVANT']]),list(Outputs[PARAMETER['RELEVANT']]), c = 'g', marker='+', label='Relevant points')
    plt.title('Predictive st. deviation around the predictions', fontsize='small')
    plt.legend(fontsize='small',loc=2)
    #Show the inferred weights with the uncertainty in the weights
    
    ax = plt.subplot(324)
    plt.plot(PARAMETER['RELEVANT'],w_infer[PARAMETER['RELEVANT']], 'bo', markersize=2)
    plt.errorbar(PARAMETER['RELEVANT'], w_infer[PARAMETER['RELEVANT']], yerr=w_std,fmt='none', ecolor='g', capsize=5, elinewidth=0.5, label='1 sd')
    plt.hlines(y = 0, xmin = 0, xmax=100, color='r', linestyles='dashed', lw=1)
    t_  = 'Posterior mean weights ({:d}) with std'.format(len(PARAMETER['RELEVANT']))
    plt.title(t_,fontsize='small')
    plt.xticks(PARAMETER['RELEVANT'], rotation=90, fontsize='x-small')
    ax.set_xticks(PARAMETER['RELEVANT'],minor=True)
    ax.grid(which='minor')
    plt.legend(fontsize='x-small')

    # Show the selected Basis functions
    
    plt.subplot(325)
    plt.plot(X,BASIS,color='k',alpha=0.2)
    plt.plot(X,BASIS[:,PARAMETER['RELEVANT']], color='m')
    plt.title('Relevant Basis (|w| > 0)', fontsize='small')
    
    #Show the S and Q factors
    
    plt.subplot(326)
    plt.stem(DIAGNOSTIC['S_FACTOR'], label='Sparsity Factor', linefmt='b-', markerfmt='bo')
    plt.stem(DIAGNOSTIC['Q_FACTOR'], label='Quality Factor', linefmt='r-', markerfmt='ro')
    plt.legend(fontsize='small')
    plt.title('Diagnostic',fontsize='small')
    
    print(likelihood)
    
#    # Analysis of S_FACTOR and Q_FACTOR
#    
#    plt.figure()
#    plt.subplot(211)
#    plt.hist(DIAGNOSTIC['S_FACTOR'],bins=100, alpha=0.5, label='Sparsity factor')
#    plt.legend()
#    plt.subplot(212)
#    plt.hist(DIAGNOSTIC['Q_FACTOR'],bins=100, alpha=0.5, label = 'Quality Fcator')
#    plt.legend()
#    

#    #######################################################################
#    # 
#    # --- PLOT THE RESULTS ---
#    #
#    #######################################################################
#        
#        
#    # Likelihood trace (and Gaussian noise info)
#        
#    plt.subplot(fRows,fCols,SP_LIKELY)
#    lsteps    = np.size(DIAGNOSTIC['LIKELIHOOD'])
#    plt.plot(range(0,lsteps), DIAGNOSTIC['LIKELIHOOD'], 'g-')
#    plt.xlim(0, lsteps+1)
#    plt.title('Log marginal likelihood trace',fontsize=TITLE_SIZE)
#        
#    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
#        ax    = plt.axis()
#        dx    = ax[1]-ax[0]
#        dy    = ax[3]-ax[2]
#        t_    = 'Actual noise:   {:.5f}'.format(noise)
#        plt.text(ax[0]+0.1*dx,ax[2]+0.6*dy,t_,fontname='Courier')
#        t_    = 'Inferred noise: {:.5f}'.format( 1/math.sqrt(HYPERPARAMETER['BETA']) )
#        plt.text(ax[0]+0.1*dx,ax[2]+0.5*dy,t_,fontname='Courier')
#            
#            
#    # Compare the generative and predictive linear models
#    if dimension == 1:
#        plt.subplot(fRows,fCols,SP_LINEAR)
#            
#        if dimension == 1:
#            plt.plot(X,z,'b-', linewidth=4, label='Actual') 
#            plt.plot(X,y,'r-', linewidth=3, label='Model')
#        else:
#            pass
#        plt.title('Generative function and linear model',fontsize=TITLE_SIZE)
#        legend = plt.legend(loc=2, shadow=False, fontsize='small', frameon=False)
#    
#    
#    # Compare the data and the predictive model (post link-function)
#    if dimension == 1:
#        plt.subplot(fRows,fCols,SP_COMPARE)
#        if dimension == 1:
#            plt.plot(X,Outputs,'k.', linewidth=4)
#            plt.plot(X,y_l,'r-', linewidth=3)
#            plt.plot(X[PARAMETER['RELEVANT']],Outputs[PARAMETER['RELEVANT']],'yo', markersize=8, clip_on=False)
#        else:
#            pass
#        plt.title('Data and predictor',fontsize=TITLE_SIZE)
#    
#    
#    # Show the inferred weights
#        
#    plt.subplot(fRows,fCols,SP_WEIGHTS)
#    markerline, stemlines, baseline = plt.stem(w_infer, 'r-.')
#    plt.setp(markerline, markerfacecolor='r')
#    plt.setp(baseline, color='k', linewidth=1)
#    plt.xlim(0, N+1)
#    t_    = 'Inferred weights ({:d})'.format(len(PARAMETER['RELEVANT']))
#    plt.title(t_,fontsize=TITLE_SIZE)
#    
#    
#    # Show the "well-determinedness" factors
#        
#    plt.subplot(fRows,fCols,SP_GAMMA)
#    ind = np.arange(len(DIAGNOSTIC['GAMMA'])) + 1
#    plt.bar(ind, DIAGNOSTIC['GAMMA'], 0.7, color='g', align = 'center')
#    plt.xlim(0, len(PARAMETER['RELEVANT'])+1)
#    plt.ylim(0, 1.1)
#    plt.title('Well-determinedness (gamma)',fontsize=TITLE_SIZE)


def get_position_consecutive_flips(a):
    
    positions = []
    for i in np.arange(1,len(a)):
        if ~((a[i-1]*a[i])>0):
            #print ((a[i-1], a[i]))
            positions.append((i-1,i))
    return positions

def get_abscissa_stationary(X,positions):
    
    approx_critical_points = []
    for (i,j) in positions:
        center_point = np.mean((X[i], X[j]))
        approx_critical_points.append(center_point)
    return approx_critical_points

    #deltas = np.diff(Outputs.T).T
    #delta_o = np.gradient(np.array(Outputs).reshape(N,))
   # delta_z = np.gradient(np.array(z).reshape(N,))
    #rolling_var_o = pd.rolling_var(delta_o, 7, min_periods=1, ddof=1)
#    rolling_var_z = pd.rolling_var(delta_z, 7, min_periods=1, ddof=1)
#    
#    plt.plot(X,Outputs,'ko',markersize=2)
#    plt.plot(X,z)
#    
#    plt.plot(X,delta_o)
#    plt.plot(X,delta_z)
#    plt.plot(X[1:], rolling_var_z)
    
    #X = np.sort(np.random.uniform(-2,4,100))
    X = np.linspace(-2,4,100)
    N = 100
    
    x = np.multiply(np.sin(np.power(X,2)),X)
    
    # First and second derivatives
    y_first = np.gradient(np.array(x).reshape(N,))
    y_second = np.gradient(y_first)
    
    # Variance of the gradient
    
    rvar = pd.rolling_var(y_first, 5, min_periods=1, ddof=1, center=True)
    
    plt.figure(figsize=(10,8))
    plt.plot(X, x, 'b-', label='True function (no noise)')
    plt.plot(X,x, 'go', markersize=4)
    plt.plot(X,y_first,  color='orange', label='Gradient')
    plt.plot(X, y_second, color='cyan', label='Second-order/Curvature')
    plt.axhline(y=0, color='r')
    plt.plot(X,rvar, color='m', label='Rolling Variance')

    critical_points = get_abscissa_stationary(X, positions)

    # Plot critical points
    plt.vlines(x=np.array(critical_points), ymin=min(x) - 1, ymax= max(x) + 1, linestyle='--', alpha=0.4)
    plt.legend(fontsize='x-small')
    
    plt.title('Placement of Basis functions in the 1d case', fontsize='x-small')
    