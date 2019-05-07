#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:50:53 2019

@author: vidhi

Synthetic data generating script

"""

import numpy as np
import pymc3 as pm
import scipy.stats as st

def generate_gp_latent(X_all, mean, cov):
    
    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval())

def generate_gp_training(X_all, f_all, n_train, noise_sd, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.3*st.norm.pdf(X_all, 2, 1) + 0.4*st.norm.pdf(X_all, 12, 1) + 0.3*st.norm.pdf(X_all, 27, 2)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
     
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    X = X_all[train_index]
    f = f_all[train_index]
    X_star = X_all[test_index]
    f_star = f_all[test_index]
    y = f + np.random.normal(0, scale=noise_sd, size=n_train)
    return X, y, X_star, f_star, f

def generate_fixed_domain_training_sets(X_all, f_all, noise_sd, uniform, seq_n_train, seq_delta):
      
      data_sets = {}
      
      X = np.empty(shape=(0, 1))
      X_star = np.empty(shape=(0,1))
      y = np.empty(shape=(0, ))
      f  = np.empty(shape=(0,))
      f_star = np.empty(shape=(0, ))
      
      for (i,j) in zip(seq_n_train, seq_delta):
            
            X_add, y_add, X_star, f_star, f_add = generate_gp_training(X_all, f_all, j, noise_sd, uniform)
            
            X = np.vstack((X, X_add))
            y = np.concatenate((y, y_add))
            f = np.concatenate((f, f_add))
            
            data_sets.update({'X_' + str(i): X})
            data_sets.update({'y_' + str(i): y})
            data_sets.update({'f_' + str(i): f})
            data_sets.update({'X_star_' + str(i): X_star})
            data_sets.update({'f_star_' + str(i): f_star})
            
      return data_sets

def generate_increasing_domain_training_sets():
      
      return
            
def persist_datasets(X, y, X_star, f_star, path, suffix):
      
     print('Saving in ' + path)
     X.tofile(path + 'X' + suffix + '.csv', sep=',')
     X_star.tofile(path + 'X_star' + suffix + '.csv', sep=',')
     y.tofile(path + 'y' +  suffix + '.csv', sep=',')
     f_star.tofile(path + 'f_star' + suffix + '.csv', sep=',')
     
def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')
    

def plot_datasets(data_sets, snr, unif):
      
      plt.figure(figsize=(20,5))
      
      for i, j in zip([0,1,2,3], seq_n_train):
            plt.subplot(1,4,i+1)
            plt.plot(data_sets['X_star_' + str(j)], data_sets['f_star_' + str(j)], "dodgerblue", lw=3, label="True f")
            plt.plot(data_sets['X_' + str(j)], data_sets['y_' + str(j)], 'ok', ms=3, alpha=0.5, label="Data")
            plt.xlabel("X") 
            plt.ylabel("The true f(x)") 
            plt.title('N = ' + str(j), fontsize = 'small')
            plt.ylim(-25, 25)
      plt.tight_layout()
      plt.suptitle('Fixed Domain Training data, SNR = ' + str(snr) + ', ' + unif)

if __name__ == "__main__":

    n_train = 20
    n_star = 200
    
    xmin = 0
    xmax = 30
    
    X_all = np.linspace(xmin, xmax,n_star)[:,None]
    
    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyperparameters 
    
    sig_sd_true = 10.0
    lengthscale_true = 5.0
    
    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]
    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)
    
    # This will change the shape of the function
    
    f_all = generate_gp_latent(X_all, mean, cov)
    
    # Data attributes
    
    noise_sd_true = np.sqrt(10)
    
    snr = np.round(sig_sd_true**2/noise_sd_true**2)
    
    uniform = False
    
    seq_n_train = [5, 10, 20, 40]  
    seq_delta = [5, 5, 10, 20]
    
    data_sets = generate_fixed_domain_training_sets(X_all, f_all, noise_sd_true, uniform, seq_n_train)
    plot_datasets(data_sets, snr, 'NUnif')

    for i in seq_n_train:
      
          suffix = '_' + str(i)
          struct_tag = 'Unif' if uniform else 'NUnif'
         
          snr_tag = 'SNR_' + str(int(np.round(snr)))
      
          X = data_sets['X' + suffix]
          y = data_sets['y' + suffix]
          X_star = data_sets['X_star' + suffix]
          f_star = data_sets['f_star' + suffix]    
          
          path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/1d/' + struct_tag + '/' + snr_tag +'/'
          persist_datasets(X, y, X_star, f_star, path, suffix)


    
    