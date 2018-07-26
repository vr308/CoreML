#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:34:16 2018

@author: vr308
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:22:13 2018

@author: vr308
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as Ck, WhiteKernel
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pylab as plt
import matplotlib.backends.backend_pdf
#from matplotlib import animation
import numpy as np


# Sklearn version of GPs

def f(x):
    """The function to predict."""
    return np.square(x)*np.sin(x)
    #return x*np.sin(x)
    #return 0.5*np.sin(x) + 0.5*x -0.02(x-5)^2
    #return -np.cos(np.pi*x) + np.sin(4*np.pi*x)

def f2(x,y):
    """ The 2 d function to predict """
    return x*np.exp(-np.square(x)-np.square(y))


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
    
    
def plot_seq_gp():
    return

if __name__ == "__main__":
    
    X = np.atleast_2d(np.sort(np.random.uniform(0,+20,40))).T
    x = np.atleast_2d(np.linspace(0, 20, 1000)).T

    noise = np.random.normal(0, 20, len(X)).reshape(len(X),1)
    y = f(X) + noise
        
    i = 1
    bool_array = np.array([False]*len(X))
    bool_array[[30]] = True
    
    remainder_err = []
    
    delta = 3
        
    while i < len(X):
    
        print ('Iteration number ' + str(i))

        X_active = X[bool_array]
        y_active = y[bool_array]
        
        X_remainder = X[~bool_array]
        y_remainder = y[~bool_array]
        
        # Instansiate a Gaussian Process model
        kernel = Ck(1000.0, (1e-3, 1e4)) * RBF(1.0, (1, 1e2)) + WhiteKernel(noise_level=10, noise_level_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel,optimizer=None)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gpr.fit(X_active, y_active)        
        y_,  sigma_ = gpr.predict(X, return_std = True)
        y_pred_active, sigma_active = y_[bool_array], sigma_[bool_array] #gpr.predict(X_train, return_std = True)
        y_pred_remainder, sigma_remainder = y_[~bool_array], sigma_[~bool_array] #gpr.predict(X_test, return_std=True)
        
        rmse_remainder = np.round(np.sqrt(np.mean(np.square(y_pred_remainder - y_remainder))),2)

        print 'Remainder error : ' + str(rmse_remainder)
        remainder_err.append(rmse_remainder)


        if delta == 1:
            
            # Extract the knots with max sigma
            
            x_knots = get_data_max_var(sigma_, X)
            knot_id = [list(np.round(X,2)).index(s) for s in np.round(x_knots,2)]
            
        elif delta == 2:
            
            #Compute deviation from the mean
            knot_id = [np.argmax(np.abs(y_ - y))]
            y_min_array = []
            y_max_array = []
        
            for k in np.arange(len(y_)):
                y_min_array.append(np.min((y_[k],y[k])))
                y_max_array.append(np.max((y_[k],y[k])))
            
        elif delta == 3:
            
            factor = sigma_ + np.ravel(np.abs(y_ - y)) 
            knot_id = [np.argmax(factor)]
            
            y_min_factor_array = []
            y_max_factor_array = []
        
            draw_factor = sigma_ + np.abs(np.ravel(y - y_))

            for k in np.arange(len(y_)):
                y_min_factor_array.append(np.min((y_[k],draw_factor[k])))
                y_max_factor_array.append(np.max((y_[k],draw_factor[k])))
        
        print(knot_id)
        x_knots = X[knot_id]
        
        y_pred, sigma = gpr.predict(x, return_std = True)
        y_pred_knot, sigma_knots = gpr.predict(x_knots, return_std=True)
        
        #plt.stem(draw_factor, label=str(i))
    
        plot_factor = np.mod(i,8)
        
        if(plot_factor == 1):
            plt.figure(figsize=(15,8))
        
        if (plot_factor == 0):
            plt.subplot(2, 4, 8)
        else:
            plt.subplot(2, 4, np.mod(i,8))

        #plt.figure()
        plt.plot(X_active, y_active, 'bo', label='Active set ' + r'$\mathbf{y}(I_{t})$')
        plt.plot(x_knots, y_pred_knot, 'ms', markersize=5, label='Next active point')
        plt.plot(X, y, 'ro', markersize=1, label='Noisy data')
        plt.plot(x, f(x), 'r', alpha=0.3, label='True function')
        plt.plot(x, y_pred, label='Mean ' + r'$\mu_{t}$')
        plt.fill_between(np.ravel(x), np.ravel(y_pred) - 2*sigma, np.ravel(y_pred) + 2*sigma, alpha=0.2, color='c', label=r'$2 \sqrt{diag(\Sigma_{t})}$')
        if delta == 2:
            plt.vlines(np.ravel(X), ymin=y_min_array, ymax=y_max_array,alpha=0.7, color='y', label='$|\mu_{t} - y(R_{t})|$')
        elif delta == 3:
            plt.errorbar(np.ravel(X_remainder), y_remainder, yerr=factor[~bool_array] , fmt='none', ecolor='orange')
            #plt.vlines(np.ravel(X), ymin=y_min_factor_array, ymax=y_max_factor_array, alpha=0.7, color='y', label=r'$\sqrt{diag(\Sigma_{t})} + |\mu_{t} - y(R_{t})|$')
        plt.title('GPR with progressive training ' + '[Stage t = ' + str(len(X_active)) + ']' + '\n' + r'$RMSE (X(R_{t})): $' +  str(rmse_remainder), fontsize='x-small')
        #plt.legend(fontsize='x-small')
        if(np.mod(i,8) == 1):
            plt.legend(fontsize = 'x-small')
            
        if i > 1:
            delta_error = remainder_err[-2] - remainder_err[-1]
            if (delta_error < 0.05*(np.max(X) - np.min(X)) and delta_error > 0):
                
                print('Training has converged')
                print('Conducting Hyper-parameter optimization')
                print ('Initial Kernel: ' + str(gpr.kernel_))
                
                rmse_test = np.round(np.sqrt(np.mean(np.square(y_pred - f(x)))),2)
                print ('Test error on hold out set: ' + str(rmse_test))
                
                #gpr = GaussianProcessRegressor(kernel=kernel)
                #gpr.fit(X_active, y_active)
                
                #print('Optimised Kernel: ' + str(gpr.kernel_))
                
                #y_pred_opt, sigma = gpr.predict(x, return_std=True)
                   
                # Plot the evolution of test error
#                plot_factor = np.mod(i,8)
#                if(plot_factor == 0):
#                    plt.figure(figsize=(15,8))
#                    plt.subplot(2, 4, np.mod(i+1,8))  
#                else:
#                    plt.subplot(2, 4, np.mod(i+1,8))
#                    
#                #plt.subplot(2, 4, np.mod(i+1,8))  
#                #plt.figure()
#                plt.plot(remainder_err, label='Test error')
#                plt.legend(fontsize = 'x-small')
#                plt.title('Evolution of Remainder Error ' + str(len(X_active)) + ' active points ', fontsize='small')
                
                #if (np.mod(i+2,8) == 0):
                #     last_plot = 8
                #else:
                #    last_plot = np.mod(i+2,8)
                
#                plt.subplot(2, 4, last_plot)
#                plt.plot(X_train, y_train, 'bo', label='Training knots')
#                plt.plot(X, y, 'ro', markersize=1, label='Noisy data')
#                plt.plot(x, f(x), 'r', alpha=0.3, label='True function')
#                plt.plot(x, y_pred, label='Mean Prediction')
#                plt.fill_between(np.ravel(x), np.ravel(y_pred) - 1.96*sigma, np.ravel(y_pred) + 1.96*sigma, alpha=0.2, color='c', label='$\sigma^{*}$')
#                plt.title('n = ' + str(len(X_train)) + '\n' + 'RMSE Test: ' +  str(rmse_test) + '\n' + 'Hyp opt: ' + str(np.bool(gpr.optimizer)), fontsize='x-small')
#                plt.suptitle('GPR with greedy training ' + '\n' +  'Optimised Kernel: ' + str(gpr.kernel_), fontsize='x-small')
#                if(np.mod(i,8) == 1):
#                    plt.legend(fontsize = 'x-small')                        
                break;           
        bool_array[knot_id] = True
        i = i + 1
            
# Testing out on hold out set

gpr = GaussianProcessRegressor(kernel=kernel)

#Full GP

gpr_full = gpr.fit(X,y) 
y_pred_full, sigma_full = gpr_full.predict(x, return_std = True)

# Random subset GP 

gpr_subset = gpr.fit()
 

y_pred_pgp, sigma = gpr.predict(x, return_std=True)
test_err_pgp = np.sqrt(np.square(f(x) - y_pred_pgp))




        

#pdf = matplotlib.backends.backend_pdf.PdfPages("/home/raid/vr308/PHD/Code/Explorations/GP/Exp6.pdf")
#for fig in xrange(1, plt.get_fignums()[-1] +1): ## will open an empty extra figure :(
#    pdf.savefig(fig)
#pdf.close()
#plt.close('all')
