#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:56:10 2019

@author: vidhi

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

def generate_gp_latent(X_all, mean, cov):
    
    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval())

def generate_gp_training(X_all, f_all, n_train, knots, sds, noise_sd, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.3*st.norm.pdf(X_all, knots[0], sds[0]) + 0.4*st.norm.pdf(X_all, knots[1], sds[1]) + 0.3*st.norm.pdf(X_all, knots[2], sds[2])
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
        
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    X = X_all[train_index]
    f = f_all[train_index]
    y = f + np.random.normal(0, scale=noise_sd, size=n_train)
    return X, y, f, train_index

def generate_gp_test(X_all, f_all, X):
      
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    print(len(test_index))
    return X_all[test_index], f_all[test_index]


def generate_fixed_domain_data(X_all, f_all, knots, sds, noise_sd, uniform, seq_n_train):
      
       seq_n_train_r = np.flip(seq_n_train)
       data_sets = {}
       
       for i,j in zip(seq_n_train_r, [0,1,2,3]):
             
            if j == 0:
                  X, y, f, train_index = generate_gp_training(X_all, f_all, np.max(seq_n_train), knots, sds, noise_sd, uniform)
                  X_star, f_star = generate_gp_test(X_all, f_all, X)
                  core_set = pd.DataFrame(zip(X.flatten(), y, f), columns=['X','y','f'])
            else:
                  prev_n_train = seq_n_train_r[j-1]
                  sub_index = np.random.choice(np.array(core_set.index), i, replace=False)
                  core_set = core_set.loc[sub_index] 
                  
                  X = np.array(core_set['X']).reshape(i,1)
                  X_star, f_star = generate_gp_test(X_all, f_all, X)
            
            data_sets.update({'X_' + str(i): X})
            data_sets.update({'y_' + str(i): np.array(core_set['y'])})
            data_sets.update({'f_' + str(i): np.array(core_set['f'])})
            data_sets.update({'X_star_' + str(i): X_star})
            data_sets.update({'f_star_' + str(i): f_star})
            
       return data_sets
 
def get_ml_report(X, y, X_star, f_star):
      
          kernel = Ck(1, (1e-10, 1e6)) * RBF(0.1, length_scale_bounds=(1e-10, 1e6)) + WhiteKernel(0.0001, noise_level_bounds=(1e-10,1e6))
          
          gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
              
          # Fit to data 
          gpr.fit(X, y)        
          ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
          post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
          post_std = np.sqrt(np.diag(post_cov))
          post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
          post_samples = np.random.multivariate_normal(post_mean, post_cov , 10)
          rmse_ = rmse(post_mean, f_star)
          lpd_ = -log_predictive_density(f_star, post_mean, post_std)
          title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + '-LPD: ' + str(lpd_)     
          ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
                            'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
          return gpr, post_mean, post_std, post_std_nf, rmse_, lpd_, ml_deltas_dict, title


def plot_gp(X_star, f_star, X, y, post_mean, post_std, post_samples, title):
    
    plt.figure()
    if post_samples != []:
          plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
    plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3, alpha=0.5)
    plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), post_mean - 1.96*post_std, 
                     post_mean + 1.96*post_std, alpha=0.2, color='g',
                     label='95% CR')
    plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')
    

def plot_lml_surface_3way(gpr, sig_sd, lengthscale, noise_sd):
    
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    l_log = np.logspace(-5, 5, 100)
    noise_log  = np.logspace(-6, 6, 100)
    l_log_mesh, noise_log_mesh = np.meshgrid(l_log, noise_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(sig_sd), l_log_mesh[i, j], np.square(noise_log_mesh[i, j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, noise_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, noise_sd, 'rx')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise sd")
    
    plt.subplot(132)
    l_log = np.logspace(-5, 4, 100)
    signal_log  = np.logspace(-5, 4, 100)
    l_log_mesh, signal_log_mesh = np.meshgrid(l_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), l_log_mesh[i, j], np.square(noise_sd)]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Signal sd")
    
    plt.subplot(133)
    noise_log = np.logspace(-5, 2, 100)
    signal_log  = np.logspace(-5, 2, 100)
    noise_log_mesh, signal_log_mesh = np.meshgrid(noise_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), lengthscale, np.square(noise_log_mesh[i,j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(noise_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(noise_sd, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise sd")
    plt.ylabel("Signal sd")    
    plt.suptitle('LML Surface ' + '\n' + str(gpr.kernel_), fontsize='small')
    

def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')
 
if __name__ == "__main__":

    n_star = 1000
    
    xmin = 0
    xmax = 5
    
    knots = [0.5,4,4]
    sds=[0.5, 0.5, 0.5]
    X_all = np.linspace(xmin, xmax,n_star)[:,None]
    
    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyperparameters 
    
    sig_sd_true = 10.0
    lengthscale_true = 100
    
    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]
    
    cov_se = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)
    cov_se2 = pm.gp.cov.Constant(sig_sd_true**2)*(pm.gp.cov.ExpQuad(1, 100) + pm.gp.cov.ExpQuad(1, 20) + pm.gp.cov.ExpQuad(1, 40))
    cov_per = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.Periodic(1, period=40, ls=10)*pm.gp.cov.ExpQuad(1, 100)

    # This will change the shape of the function
    
    #f_all = generate_gp_latent(X_all, mean, cov_per)
    f_all = 20* np.sin(3*X_all[:, 0])
    
    # Data attributes
    
    noise_sd_true = np.sqrt(50)
    
    snr = np.round(sig_sd_true**2/noise_sd_true**2)
    
    uniform = False
    
    seq_n_train = [5, 10, 20, 40]  
    
    data_sets = generate_fixed_domain_data(X_all, f_all, knots, sds, noise_sd_true, uniform, seq_n_train)
    plot_datasets(data_sets, snr, 'NUnif')
    
    X = data_sets['X_5']
    X_star = data_sets['X_star_5']
    y = data_sets['y_5']
    f_star = data_sets['f_star_5']
    
    gpr, post_mean, post_std, post_std_nf, rmse_, lpd_, ml_deltas_dict, title = get_ml_report(X, y, X_star, f_star)
    plot_lml_surface_3way(gpr, ml_deltas_dict['sig_sd'], ml_deltas_dict['ls'], ml_deltas_dict['noise_sd'])
    plot_gp(X_star, f_star, X, y, post_mean, post_std, [], title)
    
    