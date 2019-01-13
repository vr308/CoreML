#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:03:05 2018

@author: vr308
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from matplotlib import cm
import matplotlib.pylab as plt
import itertools as it
from matplotlib.colors import LogNorm
import numpy as np

# Sklearn version of GPs

def f(x):
    """The function to predict."""
    return x*np.sin(x) #- x*np.cos(x) 

def f2(x,y):
    """ The 2 d function to predict """
    return x*np.exp(-np.square(x)-np.square(y))

def f2_(X_):
    """ The 2 d function to predict """
    x = X_[:,0]
    y = X_[:,1]
    return x*np.exp(-np.square(x)-np.square(y))

if __name__ == "__main__":
        
    # GPR in 1d
    
    X_train = np.atleast_2d(np.sort(np.random.uniform(0,20,10))).T
    X_test = np.atleast_2d(np.linspace(0, 20, 1000)).T
    
    y_train = f(X_train) + 0.1*np.std(f(X_train))*np.random.normal(0, 1, len(X_train)).reshape(len(X_train),1)
    y_test = f(X_test) +  0.1*np.std(f(X_train))*np.random.normal(0, 1, len(X_test)).reshape(len(X_test),1)
    
    # Instansiate a Gaussian Process model
    #kernel = Ck(1000.0, (1e-10, 1e3)) * RBF(2, (1, 100)) + WhiteKernel(0.1, noise_level_bounds =(1e-5, 1e2))
    kernel = Ck(100.0, (1e-10, 1e3)) * RBF(3, length_scale_bounds=(2, 5)) + WhiteKernel(0.01, noise_level_bounds=(1e-5,10))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    
    # Fit to data 
    gpr.fit(X_train, y_train)        
    
    # Predict on the test set and draw samples from the posterior
    y_pred_test, sigma = gpr.predict(X_test, return_std = True)
    posterior_samples = gpr.sample_y(X_test, 50)
    
    rmse_ = np.round(np.sqrt(np.mean(np.square(y_pred_test - y_test))),2)
    lml = np.round(gpr.log_marginal_likelihood_value_,2)
        
    plt.figure()
    plt.plot(X_test, f(X_test), 'r:', label=u'$f(x) = x\sin(x)$')
    plt.plot(X_train, y_train, 'k.', markersize=8, label=u'Observations')
    plt.plot(X_test, y_pred_test, 'b-', label=u'Prediction')
    plt.fill_between(np.ravel(X_test), np.ravel(y_pred_test) - 1.96*sigma, np.ravel(y_pred_test) + 1.96*sigma, alpha=0.2, color='k')
    plt.title('GPR in action [n = 10]' + '\n' + str(gpr.kernel_) + '\n' + 'Min value of log marginal likelihood: ' +  str(lml), fontsize='small')
    plt.legend(fontsize='small')
    
    # Plot the LML Surface ###############################
    
    lengthscale = np.logspace(-2,3,50)
    noise_variance = np.logspace(-4,2,50)
    l, n = np.meshgrid(lengthscale, noise_variance)
    ln = np.array(list(it.product(lengthscale,noise_variance)))
    lml_surface = []
    for i in range(2500):
        lml_surface.append(gpr.log_marginal_likelihood(([np.log(248*248), np.log(ln[i][0]), np.log(ln[i][1])])))
    
    lml = np.array(lml_surface).reshape(50,50).T
    vmin, vmax = (-lml).min(), (-lml).max() 
    vmax = 10000
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 2000), decimals=1)
    
    plt.figure()
    plt.contourf(l,n, -lml, levels=level, cmap=cm.get_cmap('jet'),norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.plot(np.exp(gpr.kernel_.theta[1]), np.exp(gpr.kernel_.theta[2]), 'r+', label='LML Local Minimum')
    plt.colorbar(format='%.1f')
    plt.xscale("log")
    plt.yscale("log")
    
        
    # GPR in 2d 
    
    x = y = np.linspace(-2,2,200)
    X, Y = np.meshgrid(x,y)
    Z = f2(X,Y)
    
    # Plotting the true surface
    
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.scatter(x_sample,y_sample,z_sample, c='k', depthshade=False)
    ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('jet'), alpha=0.5)
    plt.title(r'$xe^{-x^2-y^2}$')
    
    #Plotting sample paths from the prior
    
    x_sample = np.sort(x[0::20])
    y_sample = np.sort(y[0::20])
    z_sample = f2(x_sample,y_sample) + 0.2*np.random.randn(25)
    
    kernel = Ck(1000.0, (1e-10, 1e3)) * RBF(1, (0.001, 100)) #+ WhiteKernel(2, noise_level_bounds =(1e-5, 1e2))
    #kernel = Ck(1000, (1e-10, 1e3))* Matern(1, (0.001, 100))
    gpr = GaussianProcessRegressor(kernel=kernel,alpha=0.001,optimizer=None)
    
    X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
    X_ = np.array(list(it.product(x_sample,y_sample)))
    
    Z_samples = gpr.sample_y(X_,7)
    
    fig = plt.figure()
    
    for i in np.arange(1, 7):
        ax = fig.add_subplot(2,3, i, projection='3d')
        ax.plot_surface(X_sample,Y_sample,Z_samples[:,i].reshape(25,25), rstride=1, cstride=1, cmap=cm.get_cmap('jet'), alpha=0.5)
    plt.suptitle('Sample paths from a 2d GP Prior with kernel ' + str(gpr.kernel), fontsize='x-small')

    # Declaring training and test sets (ensuring they dont overlap)
    
    XY_train = X_[0::47]
    Z_train = f2_(XY_train)
    
    ntrain = len(Z_train)
    
    XY_test = X_
    Z_test = f2(X_sample, Y_sample)
    
    gpr.fit(XY_train, Z_train)     
    
    # Predict on the test set
    Z_pred_test, sigma = gpr.predict(XY_test, return_std = True)
    
    rmse_ = np.round(np.sqrt(np.mean(np.square(Z_pred_test - Z_test))),2)
    lml = np.round(gpr.log_marginal_likelihood_value_,2)
     
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')    
    ax.plot_surface(X, Y, Z, label=u'$f(x) = xe^{-x^{2}-y^{2}}$', cmap=cm.get_cmap('jet'), alpha=0.5)
    ax.plot(XY_train[:,0], XY_train[:,1], Z_train, 'ko', markersize=1,label=u'Observations')
    plt.title('True (unknown surface) w noiseless observations', fontsize='small')
    ax.set_zlim3d(-0.5,0.5)

    
    ax2 = fig.add_subplot(122, projection='3d')    
    ax2.plot(XY_train[:,0], XY_train[:,1], Z_train, 'ko', markersize=1,label=u'Observations')
    ax2.plot_surface(X_sample, Y_sample, Z_pred_test.reshape(25,25), rstride=1, cstride=1, cmap=cm.get_cmap('jet'),alpha=0.5)
    plt.title('GPR on Full training data [n = ' + str(ntrain) + ']' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LML: ' +  str(lml), fontsize='small')
    ax2.set_zlim3d(-0.5,0.5)
    plt.legend()
    

    # My trial to plot the stuff my own code
    
    lengthscale = np.logspace(-3,3,50)
    noise_variance = np.logspace(-2,0,50)
    l, n = np.meshgrid(lengthscale, noise_variance)
    ln = np.array(list(it.product(lengthscale,noise_variance)))
    lml_surface = []
    for i in range(2500):
        lml_surface.append(gpr.log_marginal_likelihood(([np.log(0.4096), np.log(ln[i][0]), np.log(ln[i][1])])))
    
    lml = np.array(lml_surface).reshape(50,50).T
    vmin, vmax = (-lml).min(), (-lml).max() 
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
    
    plt.contourf(l,n, -lml, levels=level, cmap=cm.get_cmap('jet'), alpha=0.5,norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.plot(np.exp(gpr.kernel_.theta[1]), np.exp(gpr.kernel_.theta[2]), 'r+', label='LML Local Minimum')
    plt.colorbar(format='%.1f')
    plt.xscale("log")
    plt.yscale("log")
    
    # GP Regression on real high d data
    
    import sklearn.datasets as data
    
    df = data.load_boston()
    
    train_index = range(0,400)
    test_index = range(400, 506)
    
    X_train = df.data[train_index]
    X_test = df.data[test_index]
    
    y_train = df.target[train_index]
    y_test = df.target[test_index]

    kernel = Ck(1.0, (1e-10, 1e3)) * RBF(np.ones(13), length_scale_bounds=(0.2, 1.5)) + WhiteKernel(1, noise_level_bounds=(1e-5,50))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y = True)
    
    gpr.fit(X_train, y_train)
    
    y_test_pred = gpr.predict(X_test)
    
    plt.plot(y_test)
    plt.plot(y_test_pred)
    
    err = y_test - y_test_pred
    rmse = np.sqrt(np.sum())
    

