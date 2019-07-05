k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:43:17 2019

@author: vidhi

"""

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pylab as plt
import  scipy.stats as st 
import seaborn as sns
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv

varnames = ['s_1', 'ls_2','s_3', 'ls_4','ls_5','s_6','ls_7','alpha_8','s_9','ls_10','n_11'] 

# Analytical variational inference for CO_2 data

def kernel(theta, X1, X2):
        
     # se +  sexper + rq + se 
     
     s_1 = theta[0]
     ls_2 = theta[1]
     s_3 = theta[2]
     ls_4 = theta[3]
     ls_5 = theta[4]
     s_6 = theta[5]
     ls_7 = theta[6]
     alpha_8 = theta[7]
    
     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
     dist = np.sum(X1,1).reshape(-1,1) - np.sum(X2,1)
          
     sk1 = s_1**2 * np.exp(-0.5 / ls_2**2 * sqdist)
     sk2 = s_3**2 * np.exp(-0.5 / ls_4**2 * sqdist) * np.exp((-2*np.sin(np.pi*dist)**2)*(1/ls_5**2))
     sk3 = s_6**2 * (1 + (1 / 2 * alpha_8 * ls_7**2) * sqdist)**(-alpha_8)
    
     return sk1 + sk2 + sk3

mu_theta_sub = {s_1: mu_theta['s_1'],
                ls_2:  mu_theta['ls_2'], 
                s_3: mu_theta['s_3'], 
                ls_4:  mu_theta['ls_4'],
                ls_5: mu_theta['ls_5'],
                s_6: mu_theta['s_6'],
                ls_7:  mu_theta['ls_7'], 
                alpha_8: mu_theta['alpha_8'], 
                s_9: mu_theta['s_9'], 
                ls_10: mu_theta['ls_10'], 
                n_11: mu_theta['n_11']
                }

def get_kernel_matrix_blocks(X, X_star, n_train, theta):
      
      s_1 = theta[0]
      ls_2 = theta[1]
      s_3 = theta[2]
      ls_4 = theta[3]
      ls_5 = theta[4]
      s_6 = theta[5]
      ls_7 = theta[6]
      alpha_8 = theta[7]
      s_9 = theta[8]
      ls_10 = theta[9]
      n_11 = theta[10]
  
      k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2) 
      k2 = pm.gp.cov.Constant(s_3**2)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
      k3 = pm.gp.cov.Constant(s_6**2)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
      k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11**2)
      
      cov_sig =  k1 + k2 + k3 
      cov_noise = k4
 
      K = cov_sig(X)
      K_s = cov_sig(X, X_star)
      K_ss = cov_sig(X_star, X_star)
      K_noise = K + cov_noise(X)*tt.eye(n_train)
      K_inv = np.linalg.inv(K_noise.eval())
      return K.eval(), K_s.eval(), K_ss.eval(), K_noise.eval(), K_inv

def get_empirical_covariance(trace, varnames):
      
      df = pm.trace_to_dataframe(trace)
      return pd.DataFrame(np.cov(df[varnames], rowvar=False), index=varnames, columns=varnames)

def gp_mean(theta, X, y, X_star):
      
     s_9 = theta[8]
     ls_10 = theta[9]
     n_11 = theta[10]
  
     sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(x_star**2, 1) - 2 * np.dot(X, X_star.T)
     sk4 = s_9**2 * np.exp(-0.5 / ls_10**2 * sqdist) + n_11**2
      
     K = kernel(theta, X, X)
     K_noise = K + sk4*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

def gp_cov(theta, X, y, X_star):
      
     s_9 = theta[8]
     ls_10 = theta[9]
     n_11 = theta[10]
  
     sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(x_star**2, 1) - 2 * np.dot(X, X_star.T)
     sk4 = s_9**2 * np.exp(-0.5 / ls_10**2 * sqdist) + n_11**2
      
     K = kernel(theta, X, X)
     K_noise = K + sk4*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     K_ss = kernel(theta, X_star, X_star)
     return K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), K_s)
             
dh = grad(gp_mean)
d2h = jacobian(dh)
dg = grad(gp_cov)
d2g = jacobian(dg) 

def get_vi_analytical(X, y, X_star, dh, d2h, d2g, theta, mu_theta, cov_theta):
                  
    K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), theta)     
    #K = kernel(theta, X1, X2)
    #K_noise = K + theta[8]**2 * np.exp(-0.5 / theta[9]**2 * sqdist) + theta[10]**2
    #K_inv = np.linalg.inv(K_noise)
    #K_s = kernel(theta, X1, X_star)
    #K_ss =        
    pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
    pred_vi_var =  np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))

    pred_ng_mean = []
    pred_ng_var = []
    
    pred_ng_mean = pred_vi_mean + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta)))
    pred_ng_var = pred_vi_var + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta))

    for i in np.arange(len(X_star)): # To vectorize this loop
          
          x_star = X_star[i].reshape(1,1)

          pred_ng_mean.append(pred_vi_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta))))
          print(pred_ng_mean[i])
          pred_ng_var.append(pred_vi_var[i] + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta)))

    return pred_ng_mean, pred_ng_var


if __name__ == "__main__":

      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      mac_path = '/Users/vidhi.lalchand/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      desk_home_path = '/home/vr308/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      
      path = uni_path

      data_path = path + 'Data/Co2/' 
      results_path = path + 'Results/Co2/' 
      
      # Load Co2 data
      
      df = pd.read_table(data_path + 'mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
      
      # creat a date index for the data - convert properly from the decimal year 
      
      #df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')
      
      df.dropna(inplace=True)
      
      mean_co2 = df['co2'][0]
      std_co2 = np.std(df['co2'])   
      
      # normalize co2 levels
         
      #y = normalize(df['co2'])
      y = df['co2']
      t = df['year'] - df['year'][0]
      
      sep_idx = 545
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      
      # Read in trace_fr
      
      
      
      # Read in fr_param 

      fr_df_raw = pd.read_csv(results_path + 'VI/fr_df_raw.csv', sep=',', index_col=0)
      
      mu_theta = fr_df_raw['mu_implicit'][varnames] #maybe update
      cov_theta = get_empirical_covariance(trace_fr, varnames)
      
      # Analytical mean and var
      varnames = ['s_1', 'ls_2','s_3', 'ls_4','ls_5','s_6','ls_7','alpha_8','s_9','ls_10','n_11'] 

      theta = np.array([mu_theta['s_1'], mu_theta['ls_2'], mu_theta['s_3'], mu_theta['ls_4'], mu_theta['ls_5'], mu_theta['s_6'], mu_theta['ls_7'], mu_theta['alpha_8'], mu_theta['s_9'], mu_theta['ls_10'], mu_theta['n_11']])
      
      pred_ng_mean, pred_ng_var = get_vi_analytical(t_train, y_train, t_test, dh, d2h, d2g, theta, mu_theta, cov_theta)
      
      sample_mcvi_means = pd.read_csv(results_path + 'pred_dist/means_fr.csv', sep=',')
      sample_mcvi_stds = pd.read_csv(results_path + 'pred_dist/std_fr.csv', sep=',')
      lower_fr, upper_fr = get_posterior_predictive_uncertainty_intervals(sample_mcvi_means, sample_mcvi_stds)
     
      # Plotting 
      
      plt.figure()
      plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
      plt.plot(df['year'][sep_idx:], np.mean(sample_mcvi_means), alpha=1, label='MCVI', color='g')
      plt.plot(df['year'][sep_idx:], pred_ng_mean, alpha=1, label='Det. VI', color='b')
      plt.fill_between(df['year'][sep_idx:], lower_fr, upper_fr, color='green', alpha=0.5)
      plt.fill_between(df['year'][sep_idx:], (pred_ng_mean - 1.96*np.sqrt(pred_ng_var)), (pred_ng_mean - 1.96*np.sqrt(pred_ng_var)), color='b', alpha=0.3)
      plt.plot(df['year'][sep_idx:], mu_test, alpha=1, label='Type II ML', color='r')
      plt.fill_between(df['year'][sep_idx:], (mu_test - 1.96*std_test), (mu_test + 1.96*std_test), color='red', alpha=0.3)
plt.legend(fontsize='x-small')


