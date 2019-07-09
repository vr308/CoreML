#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:32:30 2019

@author: vidhi
"""

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pylab as plt
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Span, Label, Legend
from bokeh.io import output_notebook
from bokeh.palettes import brewer
import  scipy.stats as st 
import seaborn as sns
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv

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

def get_empirical_covariance(trace_df, varnames):
      
      #df = pm.trace_to_dataframe(trace)
      return pd.DataFrame(np.cov(trace_df[varnames], rowvar=False), index=varnames, columns=varnames)

def gp_mean(theta, X, y, X_star):
      
     s_9 = theta[8]
     ls_10 = theta[9]
     n_11 = theta[10]
  
     sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
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
  
     sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk4 = s_9**2 * np.exp(-0.5 / ls_10**2 * sqdist_X) + n_11**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk4*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     K_ss = kernel(theta, X_star, X_star)
     return K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), K_s)
             
dh = elementwise_grad(gp_mean)
d2h = jacobian(dh)
dg = grad(gp_cov)
d2g = jacobian(dg) 

def get_vi_analytical(X, y, X_star, dh, d2h, d2g, theta, mu_theta, cov_theta):
                  
    #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), theta)      
    #pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
    #pred_vi_var =  np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))
    
    pred_g_mean = gp_mean(theta, X, y, X_star)
    pred_g_cov = np.diag(gp_cov(theta, X, y, X_star))

    pred_ng_mean = []
    pred_ng_var = []
    
    # To fix this 
    
    pred_ng_mean = pred_g_mean + 0.5*np.trace(np.matmul(d2h(theta, X, y, X_star), np.array(cov_theta)))
    pred_ng_var = pred_vi_var + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta))

    for i in np.arange(len(X_star)): # To vectorize this loop
          
          print(i)
          x_star = X_star[i].reshape(1,1)

          pred_ng_mean.append(pred_g_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta))))
          print(pred_ng_mean[i])
          pred_ng_var.append(pred_vi_var[i] + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta)))

    return pred_ng_mean, pred_ng_var


if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Airline/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Airline/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Results/Airline/'

      path = uni_path
      
      df = pd.read_csv(path + 'AirPassengers.csv', infer_datetime_format=True, parse_dates=True, na_values=-99.99, keep_default_na=False, dtype = {'Month': np.str,'Passengers': np.int})
      
      dates = []
      for x in df['Month']:
            dates.append(np.datetime64(x))
      
     df['Time_int'] = dates - dates[0]
     df['Time_int'] = df['Time_int'].astype('timedelta64[D]')
      
      
      y = df['Passengers']
      t = df['Time_int']
      
      sep_idx = 100
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      
      # ML 
      
      # sklearn kernel 
      
      # se +  sexper + noise
      
      sk1 = 50.0**2 * Matern(length_scale=1.0, nu = 2.5) # long term rising trend
      sk2 = 2.0**2 * RBF(length_scale=1.0) \
          * PER(length_scale=1.0, periodicity=1.0)  # seasonal component
      sk3 = 5**2 * RQ(length_scale=1.0, alpha=1.0) 
      sk4 = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, 100))  # noise terms
          
      #---------------------------------------------------------------------
          
      # Type II ML for hyp.
          
      #---------------------------------------------------------------------
          
      sk_kernel = sk1 + sk2 + sk3 + sk4
      gpr = GaussianProcessRegressor(kernel=sk_kernel, normalize_y=True, n_restarts_optimizer=300)
      
      # Fit to data 
      
      gpr.fit(t_train, y_train)
           
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f"
      % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training data")
      
      mu_fit, std_fit = gpr.predict(t_train, return_std=True)
      
      print("Predicting with trained gp on test data")
      
      mu_test, std_test = gpr.predict(t_test, return_std=True)
      
      rmse_ = np.round(np.sqrt(np.mean(np.square(mu_test - y_test))), 2)
      
      lpd_ = log_predictive_density(y_test, mu_test, std_test)
      
      plt.figure()
      plt.plot(df['Month'], df['Passengers'], 'ko', markersize=2)
      plt.plot(df['Month'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['Month'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      #plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
      plt.fill_between(df['Month'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')
      
      s_1 = np.sqrt(gpr.kernel_.k1.k1.k1.k1.constant_value)
      s_3 = np.sqrt(gpr.kernel_.k1.k1.k2.k1.k1.constant_value)
      s_6 = np.sqrt(gpr.kernel_.k1.k2.k1.constant_value)
      s_9 = np.sqrt(gpr.kernel_.k2.k1.k1.constant_value)
        
      ls_2 = gpr.kernel_.k1.k1.k1.k2.length_scale
      ls_4 = gpr.kernel_.k1.k1.k2.k1.k2.length_scale
      ls_5 =  gpr.kernel_.k1.k1.k2.k2.length_scale
      ls_7 = gpr.kernel_.k1.k2.k2.length_scale
      ls_10 = gpr.kernel_.k2.k1.k2.length_scale
        
      alpha_8 = gpr.kernel_.k1.k2.k2.alpha
      n_11 = np.sqrt(gpr.kernel_.k2.k2.noise_level)
      
      ml_deltas = {'s_1': s_1, 'ls_2': ls_2, 's_3' : s_3, 'ls_4': ls_4 , 'ls_5': ls_5 , 's_6': s_6, 'ls_7': ls_7, 'alpha_8' : alpha_8, 's_9' : s_9, 'ls_10' : ls_10, 'n_11': n_11}
      
      ml_df = pd.DataFrame(data=ml_deltas, index=['ml'])
      
      ml_df.to_csv(results_path + 'airline_ml.csv', sep=',')
      
      
     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------


with pm.Model() as airline_model:
  
      # prior on lengthscales
       
       log_l2 = pm.Uniform('log_l2', lower=-10, upper=10, testval=np.log(ml_deltas['ls_2']))
       log_l4 = pm.Uniform('log_l4', lower=-10, upper=10, testval=np.log(ml_deltas['ls_4']))
       log_l5 = pm.Uniform('log_l5', lower=-10, upper=10, testval=np.log(ml_deltas['ls_5']))
       log_l7 = pm.Uniform('log_l7', lower=-10, upper=10, testval=np.log(ml_deltas['ls_7']))
       log_l10 = pm.Uniform('log_l10', lower=-10, upper=10, testval=np.log(ml_deltas['ls_10']))

       ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
       ls_4 = pm.Deterministic('ls_4', tt.exp(log_l4))
       ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
       ls_7 = pm.Deterministic('ls_7', tt.exp(log_l7))
       ls_10 = pm.Deterministic('ls_10', tt.exp(log_l10))
       
       #ls_2 = 70
       #ls_4 = 85
       #ls_7 = 1.88
       #ls_10 = 0.1219
     
       # prior on amplitudes

       log_s1 = pm.Normal('log_s1', mu=np.log(ml_deltas['s_1']), sd=0.5)
       #log_s1 = pm.Uniform('log_s1', lower=-5, upper=7)
       log_s3 = pm.Uniform('log_s3', lower=-5, upper=10, testval=np.log(ml_deltas['s_3']))
       log_s6 = pm.Normal('log_s6', mu=np.log(ml_deltas['s_6']), sd=1)
       log_s9 = pm.Uniform('log_s9', lower=-5, upper=1, testval=np.log(ml_deltas['s_9']))

       s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_3 = pm.Deterministic('s_3', tt.exp(log_s3))
       s_6 = pm.Deterministic('s_6', tt.exp(log_s6))
       s_9 = pm.Deterministic('s_9', tt.exp(log_s9))
       
       #s_3 = 2.59
       #s_9 = 0.169
      
       # prior on alpha
      
       log_alpha8 = pm.Normal('log_alpha8', mu=np.log(ml_deltas['alpha_8']), sd=0.5)
       alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
       #alpha_8 = 0.121
       
       # prior on noise variance term
      
       log_n11 = pm.Uniform('log_n11', lower=-2, upper=5, testval=np.log(ml_deltas['n_11']))
       n_11 = pm.Deterministic('n_11', tt.exp(log_n11))
       
       #n_11 = 0.195
       
       # Specify the covariance function
       
       k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.Matern52(1, ls_2) 
       k2 = pm.gp.cov.Constant(s_3**2)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
       k3 = pm.gp.cov.Constant(s_6**2)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
       k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11)

       k =  k1 + k2 + k3
          
       gp = pm.gp.Marginal(cov_func=k)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=k4)
              
with co2_model:
      
      # HMC Nuts auto-tuning implementation

      trace_hmc = pm.sample(draws=700, tune=500, chains=1)
            
with co2_model:
    
      pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc/u_prior3/', overwrite=True)
        
with co2_model:
      
      mf = pm.ADVI()

      tracker_mf = pm.callbacks.Tracker(
      mean = mf.approx.mean.eval,    
      std = mf.approx.std.eval)
     
      mf.fit(n=40000, callbacks=[tracker_mf])
      
      trace_mf = mf.approx.sample(4000)
      
with airline_model:
      
      fr = pm.FullRankADVI()
        
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=20000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)
      
      
      