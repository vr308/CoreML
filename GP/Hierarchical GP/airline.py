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
import  scipy.stats as st 
import seaborn as sns
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa

# Analytical variational inference for Airline data

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

rv_mapping = {'s_1':  airline_model.log_s1, 
              'ls_2': airline_model.log_l2_interval__, 
              's_3':  airline_model.log_s3_interval__,
              'ls_4': airline_model.log_l4_interval__,
              'ls_5': airline_model.log_l5_interval__,
              's_6': airline_model.log_s6,
              'ls_7': airline_model.log_l7_interval__,
              'alpha_8': airline_model.log_alpha8,
              's_9': airline_model.log_s9_interval__,
              'ls_10': airline_model.log_l10_interval__,
               'n_11': airline_model.log_n11_interval__
                    }

raw_mapping = {'log_s1':  airline_model.log_s1, 
              'log_l2_interval__': airline_model.log_l2_interval__, 
              'log_s3_interval__':  airline_model.log_s3_interval__,
              'log_l4_interval__': airline_model.log_l4_interval__,
              'log_l5_interval__': airline_model.log_l5_interval__,
              'log_s6': airline_model.log_s6,
              'log_l7_interval__': airline_model.log_l7_interval__,
              'log_alpha8': airline_model.log_alpha8,
              'log_s9_interval__': airline_model.log_s9_interval__,
              'log_l10_interval__': airline_model.log_l10_interval__,
               'log_n11_interval__': airline_model.log_n11_interval__ }


name_mapping = {'log_s1':  's_1', 
              'log_l2_interval__': 'ls_2', 
              'log_s3_interval__':  's_3',
              'log_l4_interval__': 'ls_4',
              'log_l5_interval__': 'ls_5',
              'log_s6': 's_6',
              'log_l7_interval__': 'ls_7',
              'log_alpha8': 'alpha_8',
              'log_s9_interval__': 's_9',
              'log_l10_interval__': 'ls_10',
              'log_n11_interval__': 'n_11'}

prior_mapping = {}


if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = uni_path
      
      df = pd.read_csv(path + 'AirPassengers.csv', infer_datetime_format=True, parse_dates=True, na_values=-99.99, keep_default_na=False, dtype = {'Month': np.str,'Passengers': np.int})
      
      dates = []
      for x in df['Month']:
            dates.append(np.datetime64(x))
      
      df['Time_int'] = dates - dates[0]
      df['Time_int'] = df['Time_int'].astype('timedelta64[D]')
     
      df['Year'] = pd.to_datetime(df['Month'])
      
      ctime = lambda x: (float(x.strftime("%j"))-1) / 366 + float(x.strftime("%Y"))
      df['Year'] = df['Year'].apply(ctime)
     
      
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
      
      sk1 = 50**2*Matern(length_scale=1.0)
      sk2 = 2.0**2 * RBF(length_scale=1.0) \
          * PER(length_scale=1.0, periodicity=365.0, periodicity_bounds='fixed')  # seasonal component
      sk3 = 5**2 * RQ(length_scale=1.0, alpha=1.0) 
      sk4 = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, 100))  # noise terms
          
      #---------------------------------------------------------------------
          
      # Type II ML for hyp.
          
      #---------------------------------------------------------------------
          
      sk_kernel = sk1 + sk2 + sk3 + sk4
      gpr = GaussianProcessRegressor(kernel=sk_kernel, normalize_y=True, n_restarts_optimizer=500)
      
      
      # Fit to data 
      
      gpr.fit(t_train, y_train)
           
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training data")
      
      mu_fit, std_fit = gpr.predict(t_train, return_std=True)
      
      print("Predicting with trained gp on test data")
      
      mu_test, std_test = gpr.predict(t_test, return_std=True)
      
      rmse_ = pa.rmse(mu_test, y_test)
      
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      
      plt.figure()
      plt.plot(df['Year'], df['Passengers'], 'ko', markersize=2)
      plt.plot(df['Year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')
      
      s_1 = np.sqrt(gpr.kernel_.k1.k1.k1.k1.constant_value)
      s_3 = np.sqrt(gpr.kernel_.k1.k1.k2.k1.k1.constant_value)
      s_6 = np.sqrt(gpr.kernel_.k1.k2.k1.constant_value)
      s_9 = np.sqrt(gpr.kernel_.k2.k1.k1.constant_value)
      
      p = gpr.kernel_.k1.k1.k2.k2.periodicity
        
      ls_2 = gpr.kernel_.k1.k1.k1.k2.length_scale
      
      ls_4 = gpr.kernel_.k1.k1.k2.k1.k2.length_scale
      ls_5 =  gpr.kernel_.k1.k1.k2.k2.length_scale
      ls_7 = gpr.kernel_.k1.k2.k2.length_scale
      ls_10 = gpr.kernel_.k2.k1.k2.length_scale
        
      alpha_8 = gpr.kernel_.k1.k2.k2.alpha
      n_11 = np.sqrt(gpr.kernel_.k2.k2.noise_level)
      
      ml_deltas = {'s_1': s_1, 'ls_2': ls_2, 's_3' : s_3, 'ls_4': ls_4 , 'ls_5': ls_5 , 'p': p , 's_6': s_6, 'ls_7': ls_7, 'alpha_8' : alpha_8, 's_9' : s_9, 'ls_10' : ls_10, 'n_11': n_11}
      
      ml_values = [ml_deltas[v] for v in varnames]
      
      ml_df = pd.DataFrame(data=np.column_stack((varnames, ml_values)), columns=['hyp','values'])
      
      ml_df.to_csv(results_path + 'airline_ml.csv', sep=',')
      
      ml_df = pd.read_csv(results_path + 'airline_ml.csv')
      
      ml_deltas = dict(zip(ml_df['hyp'], ml_df['values']))
      
      
     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------
     
      varnames = ['s_1', 'ls_2', 's_3', 'ls_4', 'ls_5', 's_6', 'ls_7', 'alpha_8', 's_9', 'ls_10', 'n_11']


with pm.Model() as airline_model:
  
       # prior on lengthscales
       
       log_l2 = pm.Uniform('log_l2', lower=1, upper=5)
       log_l4 = pm.Uniform('log_l4', lower=-10, upper=10)
       log_l5 = pm.Uniform('log_l5', lower=-5, upper=5)
       log_l7 = pm.Uniform('log_l7', lower=-7, upper=10)
       log_l10 = pm.Uniform('log_l10', lower=-15, upper=5)
       
       #log_l2 = pm.Normal('log_l2', mu=0, sd=50)
       #log_l4 = pm.Normal('log_l4', mu=0, sd=50)
       #log_l5 = pm.Normal('log_l5', mu=0, sd=50)
       #log_l7 = pm.Normal('log_l7', mu=0, sd=50)
       #log_l10 = pm.Normal('log_l10', mu=0, sd=50)
      
       #log_p = pm.Uniform('log_p', lower=1, upper=7)

       ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
       ls_4 = pm.Deterministic('ls_4', tt.exp(log_l4))
       ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
       ls_7 = pm.Deterministic('ls_7', tt.exp(log_l7))
       ls_10 = pm.Deterministic('ls_10', tt.exp(log_l10))
       
       #p = pm.Deterministic('p', tt.exp(log_p))
       
       p = 366
       
       #ls_2 = ml_deltas['ls_2']
       #ls_4 = ml_deltas['ls_4']
       #ls_7 = ml_deltas['ls_7']
       #ls_10 = ml_deltas['ls_10']
     
       # prior on amplitudes

       log_s1 = pm.Uniform('log_s1', lower=-10, upper=5)
       log_s3 = pm.Uniform('log_s3', lower=-10, upper=7)
       log_s6 = pm.Uniform('log_s6', lower=-10, upper=10)
       log_s9 = pm.Uniform('log_s9', lower=0.5, upper=5)

       #log_s9 = pm.Uniform('log_s9', lower=-9, upper=2, testval=np.log(ml_deltas['s_9']))

       s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_3 = pm.Deterministic('s_3', tt.exp(log_s3))
       s_6 = pm.Deterministic('s_6', tt.exp(log_s6))
       s_9 = pm.Deterministic('s_9', tt.exp(log_s9))
       
       #s_3 = 2.59
       #s_9 = 0.169
      
       # prior on alpha
      
       log_alpha8 = pm.Uniform('log_alpha8', lower=-10, upper=-2)
       alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
       #alpha_8 = 0.121
       
       # prior on noise variance term
      
       #log_n11 = pm.Normal('log_n11', mu=0, sd=10)
       log_n11 = pm.Uniform('log_n11', lower=-15, upper=10)
       n_11 = pm.Deterministic('n_11', tt.exp(log_n11))
       
       #n_11 = ml_deltas['alpha_8']
       
       # Specify the covariance function
       
       k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.Matern52(1, ls_2) 
       k2 = pm.gp.cov.Constant(s_3**2)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=p, ls=ls_5)
       k3 = pm.gp.cov.Constant(s_6**2)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
       k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11**2)
       
       gp_trend = pm.gp.Marginal(cov_func=k1)
       gp_periodic = pm.gp.Marginal(cov_func=k2)
       gp_rational = pm.gp.Marginal(cov_func=k3)
       gp_noise = pm.gp.Marginal(cov_func=k4)

       k =  k1 + k2 + k3
          
       gp = gp_trend + gp_periodic + gp_rational
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=k4)
       
              
with airline_model:
      
      # HMC NUTS auto-tuning implementation

      trace_hmc = pm.sample(draws=700, tune=500, chains=2)
      
with airline_model:
      
      prior_pred = pm.sample_prior_predictive(samples=500)
            
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
      
      fr.fit(n=50000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)
      
      bij_mf = mf.approx.groups[0].bij
mf_param = {param.name: bij_mf.rmap(param.eval())
	 for param in mf.approx.params}

      bij_fr = fr.approx.groups[0].bij
fr_param = {param.name: bij_fr.rmap(param.eval())
	 for param in fr.approx.params}

      check_mf.approx.params[0].set_value(bij_mf.map(mf_param['mu']))
      check_mf.approx.params[1].set_value(bij_mf.map(mf_param['rho']))

      # Updating with implicit values
      
      mf_param = analytical_variational_opt(airline_model, mf_param, pm.summary(trace_mf))
      fr_param = analytical_variational_opt(airline_model, fr_param, pm.summary(trace_fr))

      # Saving raw ADVI results
      
      mf_df = pd.DataFrame(mf_param)
      fr_df = pd.DataFrame(fr_param)
      
      # TRaceplots
      
      varnames = ['s_1', 'ls_5', 's_3', 's_6', 's_9', 'alpha_8']
      traceplots(trace_fr, varnames, ml_deltas)
      
      # Testing convergence
      
      convergence_report(tracker_fr, fr_param, fr.hist, 'Full Rank Convergence Report')
      
      # Writing out posterior predictive means
      
      write_posterior_predictive_samples(trace_fr, 100, t_train, y_train, t_test, results_path, 'fr')

      sample_means_fr = pd.read_csv(results_path + 'means_fr_100.csv')
      sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')
      
      lower_fr, upper_fr = get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
            
      #
      
      plt.figure()
      plt.plot(t_test, sample_means_fr.T)
      
      
      # 
      
      # HMC

      sample_means_hmc, sample_stds_hmc = write_posterior_predictive_samples(trace_hmc, 20, t_test, results_path + 'pred_dist/', method='hmc') 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc.csv')
      
      mu_hmc = get_posterior_predictive_mean(sample_means_hmc)
      lower_hmc, upper_hmc = get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
            
            