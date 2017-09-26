#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:45:14 2017

@author: vidhi.lalchand
"""

import numpy as np
import matplotlib.pyplot as plt
from SB2_Likelihoods import SB2_Likelihoods
from SB2_UserOptions import SB2_UserOptions
from SB2_ParameterSettings import SB2_ParameterSettings
from SparseBayes import SparseBayes
from SB2_Sigmoid import SB2_Sigmoid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#######################################################################
#
# Support function to compute basis
#
#######################################################################

def univariate_spline_kernel(x_m,x_n):
    
    min_x = np.minimum(x_m,x_n)
    return 1 + x_m*x_n + x_m*x_n*min_x - (x_m+x_n)*np.power(min_x,2)/2.0 + np.power(min_x,3)/3.0

def distSquared(X,Y):
    nx = np.size(X, 0)
    ny = np.size(Y, 0)
    D2 =  ( np.multiply(X, X).sum(1)  * np.ones((1, ny)) ) + ( np.ones((nx, 1)) * np.multiply(Y, Y).sum(1).T  ) - 2*X*Y.T
    return D2
    
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
    
    
if __name__ == "__main__":

    rseed = 1
    np.random.seed(rseed)
    
    N = 100
    dimension = 1
    noiseToSignal = 0.2
    X = np.matrix(np.sort(np.random.uniform(-10,10,N))).T # Sampling at random locations
    C = np.matrix(np.linspace(-10,10,100)).T # Sampling at uniform intervals
    
    C3 = np.matrix(np.linspace(-10,10,80).reshape((40, 2)))
    
    a = np.sort(np.linspace(-10,10,40))
    b = np.sort(np.linspace(-10,10,40))
    X4 = np.matrix(np.column_stack((a,b))) 
    C3 = np.matrix(np.mgrid[-10:10:0.5, -10:10:0.5].reshape(2,-1).T)
    
    z = np.sin(X)/X
    noise = np.std(z, ddof=1) * noiseToSignal
    Outputs = z + noise*np.random.randn(N,1)
        

    #################################################################
    # Construction of Basis 
    ################################################################

 
    def generate_gaussian_basis(X,Y, bw): 
        return np.matrix(np.exp(-distSquared(X,Y)/(bw**2)))
        
        
    bw1 = 0.5
    bw2 = 1
    bw3 = 1.7
    
    Xa = np.asarray(X)
    X1 = np.asarray(np.ones(shape=(np.size(Xa),1)))
    Xa2 = Xa*Xa
    Xa3 = Xa2*Xa
    Xa4 = Xa3*Xa
    Xa5 = Xa3*Xa2
    Xa6 = Xa5*Xa
    Xa7 = Xa5*Xa2
    BASIS0 = np.matrix(np.column_stack((X1,Xa,Xa2,Xa3,Xa4,Xa5,Xa6,Xa7)))
    BASIS1 = np.matrix(np.exp(-distSquared(X,X)/(bw3**2)))
    BASIS2 = np.matrix(np.exp(-distSquared(C,C)/(2**2)))
    BASIS3 = np.matrix(np.exp(-distSquared(X,X)/(bw3**2)))
    
    # Visualizing different basis matrices
    
    plt.figure(figsize=(10,8))
    plt.subplot(311)
    plt.plot(X,generate_gaussian_basis(X,X,0.7).T)
    plt.xlim(-10,+10)
    plt.ylim(0,1)
    plt.title('basis width = 0.7')
    
    plt.subplot(312)
    plt.plot(X,generate_gaussian_basis(X,X,1.5))
    plt.xlim(-10,+10)
    plt.ylim(0,1)
    plt.title('basis width = 1')
    
    plt.subplot(313)
    plt.plot(X,generate_gaussian_basis(X,X,3))
    plt.xlim(-10,+10)
    plt.ylim(0,1)
    plt.title('basis width = 3')

    plt.suptitle('Design matrix with RBF Kernel: Effect of changing width/scale, centered on randomly selected samples')
    
    # Visualizing the RBF Kernel in 2 dimensions
    
    X, Y = np.meshgrid(C3[:,0], C3[:,1])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z1 = generate_gaussian_basis(X4,X4,3)
    Z2 = generate_gaussian_basis(C3,C3,6)
    Z3 = generate_gaussian_basis(C3,C3,9)
    
    ax.plot_surface(X,Y, Z1.T, rstride=1, cstride=1,linewidth=1, antialiased=True,
                cmap=cm.jet)
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X,Y, Z2, rstride=1, cstride=1,linewidth=1, antialiased=True,
                cmap=cm.jet)
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(X,Y, Z3, rstride=1, cstride=1,linewidth=1, antialiased=True,
                cmap=cm.jet)
    
    
    plt.xlim(-10,6)
    plt.ylim(-10,6)
    plt.zlim(0,1)

    BASIS = np.hstack((BASIS0,BASIS1))
    BASIS = polynomialBasis(X)
    
    
    M = BASIS.shape[1]
    likelihood_ = 'Gaussian'
   
    ####################################################################
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
    
    y = BASIS*w_infer            

    # Convert the output according to the likelihood (i.e. apply link function)
    
   # if LIKELIHOOD['InUse'] == LIKELIHOOD['Gaussian']:
    #    y_l     = y
#    if LIKELIHOOD['InUse'] == LIKELIHOOD['Bernoulli']:
#        y_l     = SB2_Sigmoid(y) > 0.5


    plt.figure()
    plt.subplot(321)
    plt.plot(X,z)
    plt.ylabel('sin(x)/x')
    plt.title('Generative function',fontsize='small')
    
    plt.subplot(322)
    plt.plot(X,Outputs,'ko',markersize=2)
    plt.title('Noisy Outputs/Targets [' + str(np.size(X)) + ' samples]',fontsize='small')
    
    
    plt.subplot(323)
    plt.plot(X,z,label='Generative function',linewidth=3)
    plt.plot(X,y, label='Predictive linear model',color='r')
    plt.title('Generative and predictive linear model',fontsize='small')
    #plt.legend(fontsize='small')
    
    plt.subplot(324)
    plt.plot(X,Outputs,'ko',markersize=2)
    plt.plot(X,y,label='Predictive linear model',color='r')
    #plt.plot(X[PARAMETER['RELEVANT']],Outputs[PARAMETER['RELEVANT']],'yo', markersize=8, clip_on=False)
    plt.title('Data and predictive linear model', fontsize='small')
    
       #  Show the inferred weights
        
#    plt.subplot(325)
#    markerline, stemlines, baseline = plt.stem(w_infer, 'r-.')
#    plt.setp(markerline, markerfacecolor='r')
#    plt.setp(baseline, color='k', linewidth=1)
#    plt.xlim(0, N+1)
#    t_    = 'Inferred weights ({:d})'.format(len(PARAMETER['RELEVANT']))
#    plt.title(t_,fontsize='small')
#    
#    
#    # Show the "well-determinedness" factors
#        
#    plt.subplot(326)
#    ind = np.arange(len(DIAGNOSTIC['GAMMA'])) + 1
#    plt.bar(ind, DIAGNOSTIC['GAMMA'], 0.7, color='g', align = 'center')
#    plt.xlim(0, len(PARAMETER['RELEVANT'])+1)
#    plt.ylim(0, 1.1)
#    plt.title('Well-determinedness (gamma)',fontsize='small')
#    
    
    

    #######################################################################
    # 
    # --- SET UP GRAPHING PARAMETERS ---
    # 
    #######################################################################

#    fRows       = 2
#    fCols       = 3
#      
#    SP_DATA     = 1
#    SP_LIKELY   = 2
#    SP_LINEAR   = 3
#    SP_COMPARE  = 4
#    SP_WEIGHTS  = 5
#    SP_GAMMA    = 6
#        
#    plt.figure(1)
#    plt.clf()
#    TITLE_SIZE    = 12
#        
#    if (dimension == 1):
#        plt.subplot(fRows,fCols,SP_DATA)# axisbg='k')
#        plt.plot(X,Outputs,'k.', clip_on=False)
#    else: 
#        plt.plot3(X[:,1],X[:,2],Outputs,'w.')
#        pass
#    
#    t_    = 'Generated data ({0} points)'.format(N)
#    plt.title(t_,fontsize=TITLE_SIZE)
#    
    

    #######################################################################
    # 
    # --- PLOT THE RESULTS ---
    #
    #######################################################################
        
        
    # Likelihood trace (and Gaussian noise info)
        
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
    # Compare the generative and predictive linear models
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
#            #plt.plot(X[PARAMETER['RELEVANT']],Outputs[PARAMETER['RELEVANT']],'yo', markersize=8, clip_on=False)
#        else:
#            pass
#        plt.title('Data and predictor',fontsize=TITLE_SIZE)
#    
#    
    # Show the inferred weights
        
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
    
