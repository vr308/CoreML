#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:53:52 2017

@author: vidhi.lalchand
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from SB2_Likelihoods import SB2_Likelihoods
from SB2_UserOptions import SB2_UserOptions
from SB2_ParameterSettings import SB2_ParameterSettings
from SparseBayes import SparseBayes
from SB2_Sigmoid import SB2_Sigmoid

rseed = 1
np.random.seed(rseed)

N = 1000

noiseToSignal = 0.2

# = (np.matrix(range(N)) / float(N)).T
#X = np.matrix(np.random.uniform(-np.pi,np.pi,N)).T
X = np.matrix(np.random.uniform(-10,10,N)).T


z1 = np.sin(X)
z2 = np.sin(X)/X
z3 = np.exp(X)

plt.plot(X,z1,'bo')
plt.plot(X,z2,'ro')
plt.plot(X,z3,'go')

noise = np.std(z2, ddof=1) * noiseToSignal
Outputs = z2 + noise*np.random.randn(N,1)
    
plt.figure()
plt.plot(X,z2,'bo')
plt.plot(X,Outputs,'ro')

 ##################################################################
 # Construction of Basis 
 ##################################################################
 
 Xa = np.asarray(X)
 X1 = np.asarray(np.ones((np.size(Xa),1)))
 Xa2 = Xa*Xa
 Xa3 = Xa2*Xa
 Xa4 = Xa2*Xa2
 Xa5 = Xa3*Xa2
 Xa6 = Xa3*Xa3
 Xa7 = Xa5*Xa2
 Xa8 = Xa7*Xa
 
 inf = X1 - (Xa2/6) + (Xa4/120) - (Xa6/5040) + (Xa8/40320)

 #BASIS = np.matrix(np.column_stack((Xa,Xa3,Xa5,Xa7)))
 BASIS = np.matrix(np.column_stack((X1,Xa2,Xa4,Xa6)))

    
 
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

    # Compute the inferred prediction function
    
    y           = BASIS*w_infer
    
    plt.figure()
    plt.plot(X,Outputs,'ro')
    plt.plot(X,y,'bo')

    # Convert the output according to the likelihood (i.e. apply link function)
    
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        y_l     = y
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
        y_l     = SB2_Sigmoid(y) > 0.5
        
    
    

    #######################################################################
    # 
    # --- PLOT THE RESULTS ---
    #
    #######################################################################
        
        
    # Likelihood trace (and Gaussian noise info)
        
    plt.subplot(fRows,fCols,SP_LIKELY)
    lsteps    = np.size(DIAGNOSTIC['LIKELIHOOD'])
    plt.plot(range(0,lsteps), DIAGNOSTIC['LIKELIHOOD'], 'g-')
    plt.xlim(0, lsteps+1)
    plt.title('Log marginal likelihood trace',fontsize=TITLE_SIZE)
        
    if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
        ax    = plt.axis()
        dx    = ax[1]-ax[0]
        dy    = ax[3]-ax[2]
        t_    = 'Actual noise:   {:.5f}'.format(noise)
        plt.text(ax[0]+0.1*dx,ax[2]+0.6*dy,t_,fontname='Courier')
        t_    = 'Inferred noise: {:.5f}'.format( 1/math.sqrt(HYPERPARAMETER['BETA']) )
        plt.text(ax[0]+0.1*dx,ax[2]+0.5*dy,t_,fontname='Courier')
            
            
    # Compare the generative and predictive linear models
    if dimension == 1:
        plt.subplot(fRows,fCols,SP_LINEAR)
            
        if dimension == 1:
            plt.plot(X,z,'b-', linewidth=4, label='Actual') 
            plt.plot(X,y,'r-', linewidth=3, label='Model')
        else:
            pass
        plt.title('Generative function and linear model',fontsize=TITLE_SIZE)
        legend = plt.legend(loc=2, shadow=False, fontsize='small', frameon=False)
    
    
    # Compare the data and the predictive model (post link-function)
    if dimension == 1:
        plt.subplot(fRows,fCols,SP_COMPARE)
        if dimension == 1:
            plt.plot(X,Outputs,'k.', linewidth=4)
            plt.plot(X,y_l,'r-', linewidth=3)
            plt.plot(X[PARAMETER['RELEVANT']],Outputs[PARAMETER['RELEVANT']],'yo', markersize=8, clip_on=False)
        else:
            pass
        plt.title('Data and predictor',fontsize=TITLE_SIZE)
    
    
    # Show the inferred weights
        
    plt.subplot(fRows,fCols,SP_WEIGHTS)
    markerline, stemlines, baseline = plt.stem(w_infer, 'r-.')
    plt.setp(markerline, markerfacecolor='r')
    plt.setp(baseline, color='k', linewidth=1)
    plt.xlim(0, N+1)
    t_    = 'Inferred weights ({:d})'.format(len(PARAMETER['RELEVANT']))
    plt.title(t_,fontsize=TITLE_SIZE)
    
    
    # Show the "well-determinedness" factors
        
    plt.subplot(fRows,fCols,SP_GAMMA)
    ind = np.arange(len(DIAGNOSTIC['GAMMA'])) + 1
    plt.bar(ind, DIAGNOSTIC['GAMMA'], 0.7, color='g', align = 'center')
    plt.xlim(0, len(PARAMETER['RELEVANT'])+1)
    plt.ylim(0, 1.1)
    plt.title('Well-determinedness (gamma)',fontsize=TITLE_SIZE)