k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:43:17 2019

@author: vidhi

"""

import pymc3 as pm
import pandas as pd
#import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad, jacobian, grad
import taylor_vi as tv
import theano.tensor as tt
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings("ignore")

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
     dist = np.abs(np.sum(X1,1).reshape(-1,1) - np.sum(X2,1))
               
     sk1 = s_1**2 * np.exp(-0.5 / ls_2**2 * sqdist)
     sk2 = s_3**2 * np.exp(-0.5 / ls_4**2 * sqdist) * np.exp(-2*(np.sin(np.pi*dist)/ls_5)**2)
     sk3 = s_6**2 * np.power(1.0 + 0.5*(sqdist / (alpha_8 * ls_7**2)), -1*alpha_8)
    
     return sk1 + sk2 + sk3

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
      k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11)
      
      cov_sig =  k1 + k2 + k3 
      cov_noise = k4
 
      K = cov_sig(X)
      K_s = cov_sig(X, X_star)
      K_ss = cov_sig(X_star, X_star)
      K_noise = K + cov_noise(X)*tt.eye(n_train)
      K_inv = np.linalg.inv(K_noise.eval())
      return K.eval(), K_s.eval(), K_ss.eval(), K_noise.eval(), K_inv

def get_empirical_covariance(trace_df, varnames):
      
      #df = pm.trace_to_dataframe(trace)
      return pd.DataFrame(np.cov(trace_df[varnames], rowvar=False), index=varnames, columns=varnames)

def gp_mean(theta, X, y, X_star):
      
     s_9 = theta[8]
     ls_10 = theta[9]
     n_11 = theta[10]
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk4 = s_9**2 * np.exp(-0.5 / ls_10**2 * sqdist_X) + n_11**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk4*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

def gp_cov(theta, X, y, X_star):
      
     s_9 = theta[8]
     ls_10 = theta[9]
     n_11 = theta[10]
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk4 = s_9**2 * np.exp(-0.5 / ls_10**2 * sqdist_X) + n_11**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk4*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     K_ss = kernel(theta, X_star, X_star)
     return K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), K_s)
             


if __name__ == "__main__":

      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/'
      mac_path = '/Users/vidhi.lalchand/Desktop/Workspace/CoreML/GP/Hierarchical GP/'
      desk_home_path = '/home/vr308/Desktop/Workspace/CoreML/GP/Hierarchical GP/'
      
      path = mac_path

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
      
      #trace_fr_df = pm.trace_to_dataframe(trace_fr)
      
      trace_fr_df = pd.read_csv(results_path + '/trace_fr_df.csv', sep=',')
      
      # Read in fr_param 

      fr_df_raw = pd.read_csv(results_path + 'VI/fr_df_raw.csv', sep=',', index_col=0)
      
      mu_theta = fr_df_raw['mu_implicit'][varnames] #maybe update
      cov_theta = get_empirical_covariance(trace_fr_df, varnames)
      
      # Analytical mean and var

      theta = np.array(mu_theta)
      
      dh = elementwise_grad(gp_mean)
      d2h = jacobian(dh)
      dg = grad(gp_cov)
      d2g = jacobian(dg) 
      
      mu_taylor, std_taylor = tv.get_vi_analytical(t_train, y_train, t_test, dh, d2h, d2g, theta, mu_theta, cov_theta)
      
      #mu_taylor = pred_ng_mean 
      #std_taylor = [np.sqrt(x) for x in pred_ng_var]
      
      rmse_taylor = pa.rmse(mu_taylor, y_test)
      se_rmse_taylor = pa.se_of_rmse(mu_taylor, y_test)
      lppd_mf, lpd_mf = pa.log_predictive_density(y_test, mu_taylor, std_taylor)

      print('rmse_mf:' + str(rmse_mf))
      print('se_rmse_mf:' + str(se_rmse_mf))
      print('lpd_mf:' + str(lpd_mf))
      
     
      #sample_mcvi_means = pd.read_csv(results_path + 'pred_dist/means_fr.csv', sep=',')
      #sample_mcvi_stds = pd.read_csv(results_path + 'pred_dist/std_fr.csv', sep=',')
      #lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_mcvi_means, sample_mcvi_stds)
     
      # Plotting 
      
#      plt.figure()
#      plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
#      plt.plot(df['year'][sep_idx:], np.mean(sample_mcvi_means), alpha=1, label='MCVI', color='g')
#      plt.plot(df['year'][sep_idx:], pred_ng_mean, alpha=1, label='Analytical VI', color='b')
#      plt.fill_between(df['year'][sep_idx:], lower_fr, upper_fr, color='green', alpha=0.5)
#      plt.fill_between(df['year'][sep_idx:], (pred_ng_mean - 1.96*np.sqrt(pred_ng_var)), (pred_ng_mean + 1.96*np.sqrt(pred_ng_var)), color='b', alpha=0.3)
#      plt.plot(df['year'][sep_idx:], mu_test, alpha=1, label='Type II ML', color='r')
#      plt.fill_between(df['year'][sep_idx:], (mu_test - 1.96*std_test), (mu_test + 1.96*std_test), color='red', alpha=0.3)
#      plt.legend(fontsize='x-small')
#      plt.title('Co2 - Test Predictions')


