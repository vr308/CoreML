#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:03:05 2018

@author: vr308
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as Ck, WhiteKernel
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial import distance
import matplotlib.pylab as plt
import random
import matplotlib.backends.backend_pdf
import numpy as np

# Sklearn version of GPs

def f(x):
    """The function to predict."""
    return np.square(x)*np.sin(x)

def f2(x,y):
    """ The 2 d function to predict """
    return x*np.exp(-np.square(x)-np.square(y))

if __name__ == "__main__":
        

    # GPR in 1d
    
    X_train = np.atleast_2d(np.sort(np.random.uniform(0,20,10))).T
    X_test = np.atleast_2d(np.linspace(0, 20, 1000)).T
    
    y_train = f(X_train) + np.random.normal(0, 20, len(X_train)).reshape(len(X_train),1)
    y_test = f(X_test) +  np.random.normal(0, 20, len(X_test)).reshape(len(X_test),1)
    
    # Instansiate a Gaussian Process model
    kernel = Ck(10000.0, (1e-10, 1e3)) * RBF(2, (0.001, 100)) + WhiteKernel(0.1, noise_level_bounds =(1e-5, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel,alpha=0.001, optimizer=None)
    
    # Plotting samples from the prior with variance estimate
    plt.figure(figsize=(10,7))
    plt.subplot(121)
    y_mean_prior, y_std_prior = gpr.predict(X_test, return_std=True) 
    plt.plot(X_test, gpr.sample_y(X_test,5))
    plt.plot(X_test, y_mean_prior, color='k')
    plt.fill_between(np.ravel(X_test), y_mean_prior - 1.96*y_std_prior, y_mean_prior  + 1.96*y_std_prior, color='k', alpha=0.2)
    plt.title('Sample paths from GP Prior w. 95% conf. intervals', fontsize='small')
    
    # Fit to data 
    gpr.fit(X_train, y_train)        
    
    # Predict on the test set
    y_pred_test, sigma = gpr.predict(X_test, return_std = True)
    
    rmse_ = np.round(np.sqrt(np.mean(np.square(y_pred_test - y_test))),2)
    lml = np.round(gpr.log_marginal_likelihood_value_,2)
        
    plt.subplot(122)
    plt.plot(X_test, f(X_test), 'r:', label=u'$f(x) = x^2\sin(x)$')
    plt.plot(X_train, y_train, 'r.', markersize=5, label=u'Observations')
    plt.plot(X_test, y_pred_test, 'b-', label=u'Prediction')
    plt.fill_between(np.ravel(X_test), np.ravel(y_pred_test) - 1.96*sigma, np.ravel(y_pred_test) + 1.96*sigma, alpha=0.2, color='k')
    plt.title('GPR on Full training data [n = 10]' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LML: ' +  str(lml), fontsize='small')
    plt.legend(fontsize='small')
        
    
    # GPR in 2d 
    
    x = y = np.linspace(-2,2,1000)
    X, Y = np.meshgrid(x,y)
    Z = f2(X,Y)
    
    x_sample = np.sort(random.sample(x,100))
    y_sample = np.sort(random.sample(y,100))
    z_sample = f2(x_sample,y_sample) + 0.2*np.random.randn(100)
    
    
    # Plotting the true surface
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.scatter(x_sample,y_sample,z_sample, c='k', depthshade=False)
    ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('jet'), alpha=0.5)
    
    
    kernel = Ck(1000.0, (1e-10, 1e3)) * RBF(1, (0.001, 100)) #+ WhiteKernel(2, noise_level_bounds =(1e-5, 1e2))
    kernel = Ck(1000, (1e-10, 1e3))* Matern(1, (0.001, 100))
    gpr = GaussianProcessRegressor(kernel=kernel,alpha=0.001)
    
    X_ = np.asarray((x, y)).T
    X_sample, Y_sample = np.meshgrid(x, y)
    
    Z_samples = gpr.sample_y(X_,10)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.scatter(x_sample,y_sample,z_sample, c='k', depthshade=False)
    ax.plot_surface(X_sample,Y_sample,Z_samples[:,5], cmap=cm.get_cmap('jet'), alpha=0.5)
    
    
    
    gpr.fit(X, y)        



# Animation stuff

#fig = plt.figure()
#ax = plt.axes(xlim=(0,20), ylim=(-40,40))
#line, = ax.plot([], [], lw=1, color='b')
#line2, = ax.plot([],[], 'r.', markersize=10, label=u'Observations')
##path, = ax.fill_between([],[],[],[], alpha=0.7)
#fills = ax.fill_between([],[],[],[])
#xdata, ydata = [], []
#    
#def init():
#    ax.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
#    #ax.title('Sequential GP Training (adaptive centers)', fontsize='small')
#    #ax.plot(X, y, 'r.', markersize=10, label=u'Observations')
#    return line, line2, 
#    
#def data_gen(i=0):
#    i = 0
#    while i < len(y_pred_evolutions):
#        i += 1
#        yield x, y_pred_evolutions[i-1], sigma_evolutions[i-1], X[i-1], y[i-1] 
#    
#def animate(data):
#    x_curve, y_pred_curve, sigma_curve, X, y = data
#    xdata.append(X)
#    ydata.append(y)
#    line2.set_data(xdata, ydata)
#    line.set_data(x_curve, y_pred_curve)
#    
#    y_upper = y_pred_curve + 2*sigma_curve
#    y_lower = y_pred_curve - 2*sigma_curve
#    z = y_upper - y_lower  
#    
#    cmap = cm.get_cmap('viridis')
#    x_array = x_curve.reshape(len(y_pred_curve),)
#    normalize = mpl.colors.Normalize(vmin = z.min(), vmax=z.max())
#    fills.remove()
#    for i in range(999):
#        fills = ax.fill_between([x_array[i],x_array[i+1]], y_lower[i], y_upper[i], color = cmap(normalize(z[i])), alpha=0.7)
#        
#    return line, line2, fills
#
## call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, func=animate, frames=data_gen, init_func=init,
#           repeat=False, interval=500, blit=False)
#
#plt.show()
