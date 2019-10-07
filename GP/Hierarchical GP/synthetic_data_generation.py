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
import pandas as pd
import matplotlib.pylab as plt

def generate_gp_latent(X_all, mean, cov):
    
    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval())

def generate_gp_training(X_all, f_all, n_train, noise_sd, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.3*st.norm.pdf(X_all, 2, 5) + 0.4*st.norm.pdf(X_all, 40, 5) + 0.3*st.norm.pdf(X_all, 87, 5)
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


def generate_fixed_domain_data(X_all, f_all, noise_sd, uniform, seq_n_train):
      
       seq_n_train_r = np.flip(seq_n_train)
       data_sets = {}
       
       for i,j in zip(seq_n_train_r, np.arange(len(seq_n_train))):
             
            if j == 0:
                  X, y, f, train_index = generate_gp_training(X_all, f_all, np.max(seq_n_train), noise_sd, uniform)
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


def generate_increasing_domain_training_sets():
      
      return;
            
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
      
      plt.figure(figsize=(10,5))
      
      for i, j in zip(np.arange(len(seq_n_train)), seq_n_train):
            plt.subplot(2,4,i+1)
            plt.plot(data_sets['X_star_' + str(j)], data_sets['f_star_' + str(j)], "dodgerblue", lw=3, label="True f")
            plt.plot(data_sets['X_' + str(j)], data_sets['y_' + str(j)], 'ok', ms=3, alpha=0.5, label="Data")
            plt.xlabel("X") 
            plt.ylabel("The true f(x)") 
            plt.title('N = ' + str(j), fontsize = 'small')
            plt.ylim(-50, 50)
      plt.tight_layout()
      plt.suptitle('Fixed Domain Training data, SNR = ' + str(snr) + ', ' + unif, fontsize='x-small')

if __name__ == "__main__":

    n_star = 500
    
    xmin = 0
    xmax = 100
    
    X_all = np.linspace(xmin, xmax,n_star)[:,None]
    
    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyperparameters 
    
    sig_sd_true = 25.0
    lengthscale_true = 20.0
    
    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)
    
    # This will change the shape of the function
    
    f_all = generate_gp_latent(X_all, mean, cov)
    
    # Data attributes
    
    noise_sd_true = np.sqrt(50)
    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]
    
    snr = np.round(sig_sd_true**2/noise_sd_true**2)
    
    uniform = False
    
    seq_n_train = [10, 20, 40, 60, 80, 100, 120]  
    
    #seq_n_train=[200]
    data_sets = generate_fixed_domain_data(X_all, f_all, noise_sd_true, uniform, seq_n_train)
    plot_datasets(data_sets, snr, 'Unif')

    for i in seq_n_train:
      
          suffix = '_' + str(i)
          struct_tag = 'Unif' if uniform else 'NUnif'
         
          snr_tag = 'snr_' + str(int(np.round(snr)))
      
          X = data_sets['X' + suffix]
          y = data_sets['y' + suffix]
          X_star = data_sets['X_star' + suffix]
          f_star = data_sets['f_star' + suffix]    
          
          path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/' + struct_tag + '/' + snr_tag +'/'
          persist_datasets(X, y, X_star, f_star, path, suffix)

    # Persist the f_all and X_all
      
    path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/' 
      
    np.savetxt(path + 'X_all.csv', X_all, delimiter=',')
    np.savetxt(path + 'f_all.csv', f_all, delimiter=',')
      
    # Read f_all and X_all
     
    X_all = np.array(pd.read_csv(path + 'X_all.csv', sep=',', header=None))
    f_all = np.array(pd.read_csv(path + 'f_all.csv', sep=',', header=None)).reshape(500,)
     
     

     
     
     