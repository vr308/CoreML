#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:03:05 2018

@author: vr308
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from matplotlib import cm
import matplotlib as mpl
from scipy.spatial import distance
import matplotlib.pylab as plt
import random
import matplotlib.backends.backend_pdf
#from matplotlib import animation
import numpy as np

# Sklearn version of GPs

def f(x):
    """The function to predict."""
    return np.square(x)*np.sin(x)

#def plot_gp_fit_predict(X_train, x, y_train, y_pred, sigma, knots, y_pred_knots, sigma_knots, title, gpr, entropy):
#    
#    #y_upper = y_pred.reshape(len(x),) + 2*sigma
#    #y_lower = y_pred.reshape(len(x),) - 2*sigma
#    
#    #upper_coords = np.column_stack((x, y_upper))
#    #lower_coords = np.column_stack((x, y_lower))
#
#    #dist_matrix = distance.cdist(upper_coords, lower_coords, metric='euclidean')
#
#    #z=[np.min(dist_matrix[i]) for i in np.arange(len(upper_coords))]
#    
#    #cmap = cm.get_cmap('viridis')
#    #x_array = x.reshape(len(y_pred),)
#    
#    plt.figure()
#    plt.plot(x, f(x), 'r:', label=u'$f(x) = x^2\sin(x)$')
#    plt.plot(X_train, y_train, 'r.', markersize=10, label=u'Training Observations')
#    plt.plot(x, y_pred, 'b-', label=u'Prediction')
#    plt.fill_between(x.reshape(len(x),), y_pred.reshape(len(x),) - 1.96*sigma, y_pred.reshape(len(x),) + 1.96*sigma, alpha=0.5, color='y')
#    
#    #normalize = mpl.colors.Normalize(min(sigma), max(sigma))
#    #for i in range(999):
#    #    plt.fill_between([x_array[i],x_array[i+1]], y_lower[i], y_upper[i], color = cmap(normalize(sigma[i])), alpha=0.7)
#    #for i in range(10):
#    #    plt.plot(x, gpr.sample_y(x, 10)[i][0], alpha=0.5, linewidth=0.5)
#    #if entropy:
#    #    plt.errorbar(knots, y_pred_knots, yerr=sigma_knots*2, barsabove=True, ls='None', marker='s', capsize=10, color='orange')
#    plt.xlabel('$x$')
#    plt.ylabel('$f(x)$')
#    plt.ylim(-300, +300)
#    #plt.legend(loc='upper left')
#    plt.title(title, fontsize='small')
#    #plt.annotate('Knots at ' + str(knots), xy=[5,8])
        
def get_max_entropy(sigma_, X):
    
    pos_max_sigma = np.where(np.round(sigma_,4) == np.round(np.max(sigma_),4))[0]
    if len(pos_max_sigma) > 1:
        x_ = X[pos_max_sigma].reshape(len(pos_max_sigma),)
        if np.where(np.diff(pos_max_sigma) > 1)[0].size > 0:
            print('There are more than 1 regions of max entropy, identifying the centers of each region')
            break_points = np.where(np.diff(pos_max_sigma) > 1)[0]
            regions = np.split(x_,break_points+1)
            return [np.round(np.median(k),2) for k in regions]
        else:
            print('There is 1 region of max entropy, returning its center')
            return [np.round(np.median(x[pos_max_sigma]),2)]
    else:
        print('There is one distinct point of highest variance')
        return [np.round(np.median(x[pos_max_sigma]),2)]
    
    
def get_data_max_var(sigma_, X):
      
    sigma_ = np.round(sigma_,4)
    pos_max_sigma = np.where(sigma_ == np.max(sigma_))[0]
    x_ = X[pos_max_sigma].reshape(len(pos_max_sigma),)
    if len(pos_max_sigma) > 1:
        if np.where(np.diff(pos_max_sigma) > 1)[0].size > 0:
            print('There are more than 1 regions of max entropy, identifying the centers of each region')
            break_points = np.where(np.diff(pos_max_sigma) > 1)[0]
            regions = np.split(x_,break_points+1)
            mid_index = [len(k)//2 for k in regions]
            return np.array([np.round(regions[i][mid_index[i]],4) for i in np.arange(len(mid_index))])[:,None]
        else:
            print('There is 1 region of max entropy, returning its center')
            mid_index = len(pos_max_sigma)//2
            return np.array([np.round(x_[mid_index],4)])[:,None]
    else:
        print('There is one distinct point of highest variance')
        return np.array([np.round(x_,4)]).reshape(1,1)
    
def plot_mle_trace(mle_trace, mean_mle_trace):
    
    plt.figure()
    plt.title('Negative MLE Trace', fontsize='small')
    plt.plot(mle_trace,label='Greedy Centers')
    plt.plot(mean_mle_trace, label = 'Uniform Centers')
    plt.legend(fontsize='small')

if __name__ == "__main__":
        
    #  The  noisy case # ----------------------------------------------------------------------
    
    X = np.atleast_2d(np.sort(np.random.uniform(0,20,40))).T
    x = np.atleast_2d(np.linspace(0, 20, 1000)).T

    noise = np.random.normal(0, 20, len(X)).reshape(len(X),1)
    
    y = f(X) + noise
    
    
    #random.randint(0,len(X))
    
    i = 0  
    #y_pred_evolutions = []
    #sigma_evolutions = []
    bool_array = np.array([False]*len(X))
    bool_array[[36]] = True
    
    training_err = [] 
    test_err = []
    
    while i < len(X):
    
        #dist_matrix = distance.cdist(np.atleast_2d(seq_).T, X_train)
        #closest_index = [x.argmin() for x in dist_matrix]
        
        print ('Iteration number ' + str(i))

        X_train = X[bool_array]
        y_train = y[bool_array]
        
        X_test = X[~bool_array]
        y_test = y[~bool_array]
        
        # Instansiate a Gaussian Process model
        kernel = Ck(1000.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(noise_level=100, noise_level_bounds=(1e-10, 20))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gpr.fit(X_train, y_train)        
        y_,  sigma_ = gpr.predict(X, return_std = True)
        y_pred_train, sigma_train = y_[bool_array], sigma_[bool_array] #gpr.predict(X_train, return_std = True)
        y_pred_test, sigma_test = y_[~bool_array], sigma_[~bool_array] #gpr.predict(X_test, return_std=True)
        
        rmse_train = np.round(np.sqrt(np.mean(np.square(y_pred_train - y_train))),2)
        rmse_test = np.round(np.sqrt(np.mean(np.square(y_pred_test - y_test))),2)

        print 'Training error : ' + str(rmse_train)
        print 'Test error : ' + str(rmse_test)
        
        if i > 0:
            if np.abs(test_err[-1] - rmse_test) < 1e-1:
                print('Training has converged')
                break;
                
        training_err.append(rmse_train)
        test_err.append(rmse_test)

        # Extract the knots with max sigma
        #x_knots = get_data_max_var(sigma_, X)
        #knot_id = [list(np.round(X,2)).index(s) for s in np.round(x_knots,2)]
        
        knot_id = [np.argmax(np.abs(y_ - y))]
        x_knots = X[knot_id]
        
        y_pred, sigma = gpr.predict(x, return_std = True)
        y_pred_knots, sigma_knots = gpr.predict(x_knots, return_std=True)
        
        factor = np.ravel(np.abs(y_ - y)) 
        
        #lml = np.round(gpr.log_marginal_likelihood_value_,2)
        
        #y_pred_evolutions.append(y_pred)
        #sigma_evolutions.append(sigma)
        
        y_min_array = []
        y_max_array = []
        
        for k in np.arange(len(y_)):
            y_min_array.append(np.min((y_[k],y[k])))
            y_max_array.append(np.max((y_[k],y[k])))
        
        plt.figure()
        title = 'GP Regression ' + 'Iteration ' + str(i+1)
        #plot_gp_fit_predict(X, x_test, y, y_pred, sigma, knots_, y_pred_test, sigma_knots, title, gpr, True)
        plt.plot(X_train, y_train, 'bo', label='Training knots')
        plt.plot(x_knots, y_pred_knots, 'g+', markersize=20, label='Proposed Knots')
        plt.plot(X, y, 'ro', markersize=1, label='Noisy data')
        #plt.plot(X_test, y_pred_test, 'g-', markersize=1, label= 'Test data')
        plt.plot(x, f(x), 'r', alpha=0.3, label='True function')
        plt.plot(x, y_pred, label='Mean Prediction')
        plt.fill_between(np.ravel(x), np.ravel(y_pred) - 1.96*sigma, np.ravel(y_pred) + 1.96*sigma, alpha=0.2, color='c', label='$\sigma^{*}$')
        plt.vlines(np.ravel(X), ymin=y_min_array, ymax=y_max_array,alpha=0.7, color='y', label='$|y^{*} - y|$')
        plt.title('GPR with greedy training [n = ' + str(len(X_train)) +  ']' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE Tr: ' + str(rmse_train) + '\n' + 'RMSE Test: ' +  str(rmse_test), fontsize='small')
        plt.legend(fontsize = 'x-small')
            
        bool_array[knot_id] = True
        i = i + 1

plt.figure()
plt.plot(test_err, label='Test error')
plt.legend()
plt.title('Test Error ' + str(len(X_train)) + ' training points, ' + str(i) + ' iterations ', fontsize='small')

        
pdf = matplotlib.backends.backend_pdf.PdfPages("Exp3.pdf")
for fig in xrange(1, plt.get_fignums()[-1] +1): ## will open an empty extra figure :(
    pdf.savefig(fig)
pdf.close()


# Train a GP on uniformly distributed sample in the range of X.

# Instansiate a Gaussian Process model

kernel = Ck(1000.0, (1e-10, 1e3)) * RBF(5, (0.001, 100)) + WhiteKernel(0.2, noise_level_bounds =(1e-5, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel,alpha=0.001)
# Fit to data
 
gpr.fit(X, y)        

# Predict on the full x-axis

x = np.atleast_2d(np.linspace(0, 20, 1000)).T
y_pred, sigma = gpr.predict(x, return_std = True)

rmse_ = np.round(np.sqrt(np.mean(np.square(y_pred - y)),2)
lml = np.round(gpr.log_marginal_likelihood_value_,2)


title = 'GP Regression on full training set'

plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x^2\sin(x)$')
plt.plot(X, y, 'r.', markersize=5, label=u'Observations')
#plt.plot(X, y, 'g+', markersize=10, label=u'Chosen Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill_between(x.reshape(len(x),), y_pred.reshape(len(x),) - 1.96*sigma, y_pred.reshape(len(x),) + 1.96*sigma, alpha=0.5, color='y')
plt.title('GPR on Full training data [n = 40]' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse) + '\n' + 'LML: ' +  str(lml), fontsize='small')
plt.legend(fontsize='small')

plot_gp_fit_predict(X_train, x, y_train, y_pred, sigma, None, None, None, title, gpr, False)




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
