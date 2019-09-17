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
import advi_analysis as ad

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

def get_cov_point(theta, X):
      
       k1 = pm.gp.cov.Constant(theta['s_1']**2)*pm.gp.cov.Matern52(1, theta['ls_2']) 
       k2 = pm.gp.cov.Constant(theta['s_3']**2)*pm.gp.cov.ExpQuad(1, theta['ls_4'])*pm.gp.cov.Periodic(1, period=366, ls=theta['ls_5'])
       k3 = pm.gp.cov.Constant(theta['s_6']**2)*pm.gp.cov.ExpQuad(1, theta['ls_7']) +  pm.gp.cov.WhiteNoise(theta['n_8']**2)
       
       k = k1 + k2 + k3
      
       return k(X,X)

rv_mapping = {'s_1':  airline_model.log_s1, 
              'ls_2': airline_model.log_l2, 
              'ls_3':  airline_model.log_l3,
              's_4': airline_model.log_s4,
              'ls_5': airline_model.log_l5,
              'n_6': airline_model.log_n6}
          

raw_mapping = {'log_s1_interval__':  airline_model.log_s1_interval__, 
              'log_l2_interval__': airline_model.log_l2_interval__, 
              'log_s3_interval__':  airline_model.log_s3_interval__,
              'log_l4_interval__': airline_model.log_l4_interval__,
              'log_l5_interval__': airline_model.log_l5_interval__,
              'log_s6_interval__': airline_model.log_s6_interval__,
              'log_l7_interval__': airline_model.log_l7_interval__,
              'log_alpha8_interval__': airline_model.log_alpha8_interval__,
              'log_s9_interval__': airline_model.log_s9_interval__,
              'log_l10_interval__': airline_model.log_l10_interval__,
               'log_n11_interval__': airline_model.log_n11_interval__}


name_mapping = {'log_s1_interval__':  's_1', 
              'log_l2_interval__': 'ls_2', 
              'log_s3_interval__':  's_3',
              'log_l4_interval__': 'ls_4',
              'log_l5_interval__': 'ls_5',
              'log_s6_interval__': 's_6',
              'log_l7_interval__': 'ls_7',
              'log_alpha8_interval__': 'alpha_8',
              'log_s9_interval__': 's_9',
              'log_l10_interval__': 'ls_10',
              'log_n11_interval__': 'n_11'}


def forward_mu(x, mu, std):
      
      return x*std + mu

def forward_std(x, std):
      
      return x*std


if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

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
      
      # Whiten the data 
      
      emp_mu = np.mean(y)
      emp_std = np.std(y)
      
      y =  (y - emp_mu)/emp_sd
      
      sep_idx = 100
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      
      # ML 
      
      # sklearn kernel 
      
      # se +  sexper + noise
      
      sk1 = 50**2 * Matern(length_scale=1.0, length_scale_bounds=(0.000001,1.0))
      sk2 = 20**2 * RBF(length_scale=1.0, length_scale_bounds=(1e-3,1e+7)) \
          * PER(length_scale=1.0, periodicity=365.0, periodicity_bounds='fixed')  # seasonal component
          
     # sk2 = 50**2 * Matern(length_scale=1.0, length_scale_bounds=(0.000001,5000.0))* PER(length_scale=1.0, periodicity=365.0, periodicity_bounds='fixed')  # seasonal component
      #sk3 = 5**2 * RQ(length_scale=1.0, alpha=1.0) 
      sk4 = 0.1**2 * RBF(length_scale=0.1, length_scale_bounds=(1e-3,1e3)) + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, 100))  # noise terms
      
      #---------------------------------------------------------------------
          
      # Type II ML for hyp.
          
      #---------------------------------------------------------------------
      
      varnames = ['s_1', 'ls_2', 'ls_3', 's_4', 'ls_5', 'n_6']
      
      sk_kernel =  sk2 + sk4
     
      gpr_lml = []
      gpr_ml_deltas = []
      gpr_models = []
      
      # Fit to data 
      for i in np.arange(9):
            gpr = GaussianProcessRegressor(kernel=sk_kernel, normalize_y=False, n_restarts_optimizer=10)
            print('Fitting ' + str(i))
            gpr.fit(t_train, y_train)
            gpr_models.append(gpr)
            gpr_lml.append(gpr.log_marginal_likelihood(gpr.kernel_.theta))
            gpr_ml_deltas.append(gpr.kernel_)
           
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
      mu_fit, std_fit = gpr.predict(t_train, return_std=True)      
      mu_test, std_test = gpr.predict(t_test, return_std=True)
      
      mu_fit = forward_mu(mu_fit, emp_mu, emp_std)
      std_fit = forward_std(std_fit, emp_std)
      
      mu_test = forward_mu(mu_test, emp_mu, emp_std)
      std_test = forward_std(std_test, emp_std)
      
      rmse_ = pa.rmse(mu_test, forward_mu(y_test, emp_mu, emp_std))
      
      lpd_ = pa.log_predictive_density(forward_mu(y_test, emp_mu, emp_std), mu_test, std_test)
      
      #Plotting
      
      plt.figure()
      plt.plot(df['Year'], df['Passengers'], 'ko', markersize=2)
      plt.plot(df['Year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'NLPD: ' + str(-lpd_), fontsize='small')
      
      s_1 = np.sqrt(gpr.kernel_.k1.k1.k1.constant_value)
      ls_2 = gpr.kernel_.k1.k1.k2.length_scale
      
      ls_3 = gpr.kernel_.k1.k2.length_scale
      
      s_4 = np.sqrt(gpr.kernel_.k2.k1.k1.constant_value)
      ls_5 = gpr.kernel_.k2.k1.k2.length_scale
      
      n_6 = np.sqrt(gpr.kernel_.k2.k2.noise_level)
      
      ml_deltas = {'s_1': s_1, 'ls_2': ls_2, 'ls_3' : ls_3, 's_4': s_4 , 'ls_5': ls_5 , 'n_6': n_6}
      
      ml_values = [ml_deltas[v] for v in varnames]
      
      ml_df = pd.DataFrame(data=np.column_stack((varnames, ml_values)), columns=['hyp','values'])
      
      ml_df.to_csv(results_path + 'airline_ml.csv', sep=',')
      
      # Reading what is already stored
      
      ml_df = pd.read_csv(results_path + 'airline_ml.csv')
      
      ml_deltas = dict(zip(ml_df['hyp'], ml_df['values']))
      
      
      #---------------------------------------------------------------------
          
      # Vanilla GP - pymc3
          
      #---------------------------------------------------------------------

      with pm.Model() as model:
        
             # Specify the covariance function
       
             cov_main = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2)*pm.gp.cov.Periodic(1, period=p, ls=ls_3)
             cov_noise = pm.gp.cov.Constant(s_4**2)*pm.gp.cov.ExpQuad(1, ls_5) +  pm.gp.cov.WhiteNoise(n_6**2)
             
             gp_main = pm.gp.Marginal(cov_func=cov_main)
             gp_noise = pm.gp.Marginal(cov_func=cov_noise)
      
             k =  cov_main + cov_noise
                
             gp = gp_main
            
             # Marginal Likelihood
             y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)
        
      with model:
          
            f_cond = gp.conditional("f_cond", Xnew=t_test)

      mu_ml, var_ml = gp.predict(t_test, pred_noise=True)
     
      
      plt.figure()
      plt.plot(t_test, mu_test)
      plt.plot(t_test, forward_mu(mu_ml, emp_mu, emp_std))
      
      post_pred_mean, post_pred_cov_nf = gp.predict(t_test, pred_noise=False)
      post_pred_std = np.sqrt(np.diag(post_pred_cov))
      
     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------
     
with pm.Model() as airline_model:
      
       i = pm.Normal('i', sd=1)
       c = pm.HalfNormal('c', sd=2.0)
       mean_trend = pm.gp.mean.Linear(coeffs=c, intercept=i)
      
       log_l2 = pm.Normal('log_l2', mu=0, sd=3, testval=np.log(ml_deltas['ls_2']))
       log_l3 = pm.Normal('log_l3', mu=0, sd=3, testval=np.log(ml_deltas['ls_3']))
       log_l5 = pm.Normal('log_l5', mu=np.log(ml_deltas['ls_5']), sd=1)

       ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
       ls_3 = pm.Deterministic('ls_3', tt.exp(log_l3))
       ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
       
       p = 366
       
       # prior on amplitudes

       log_s1 = pm.Normal('log_s1', mu=np.log(ml_deltas['s_1']), sd=0.4)
       log_s4 = pm.Normal('log_s4', mu=np.log(ml_deltas['s_4']), sd=3)

       #s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_4 = pm.Deterministic('s_4', tt.exp(log_s4))
             
       # prior on noise variance term
      
       log_n6 = pm.Normal('log_n6', mu=np.log(ml_deltas['n_6']), sd=1)
       n_6 = pm.Deterministic('n_6', tt.exp(log_n6))
              
       # Specify the covariance function
       
       cov_main = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2)*pm.gp.cov.Periodic(1, period=p, ls=ls_3)
       cov_noise = pm.gp.cov.Constant(s_4**2)*pm.gp.cov.ExpQuad(1, ls_5) +  pm.gp.cov.WhiteNoise(n_6**2)
       
       gp_main = pm.gp.Marginal(mean_func = mean_trend, cov_func=cov_main)
       gp_noise = pm.gp.Marginal(cov_func=cov_noise)

       k = k2 
          
       gp = gp_main
       
       trace_prior = pm.sample(1000)

with airline_model:
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)
       #prior_pred = pm.sample_ppc(trace_prior, samples=50)
       
with airline_model:
      
       trace_hmc = pm.sample(draws=2000, tune=700, chains=2)

      
# Prior predictive check
       
pa.prior_predictive(t_train, y_train, prior_pred)
                 
with airline_model:
    
      pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc_final/', overwrite=True)
      
with airline_model:
      
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/')
        
with airline_model:
      
      mf = pm.ADVI()

      tracker_mf = pm.callbacks.Tracker(
      mean = mf.approx.mean.eval,    
      std = mf.approx.std.eval)
     
      mf.fit(n=80000, callbacks=[tracker_mf])
      
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

      # Updating with implicit values - %TODO Testing
      
      mf_param = ad.analytical_variational_opt(airline_model, mf_param, pm.summary(trace_mf), raw_mapping, name_mapping)
      fr_param = ad.analytical_variational_opt(airline_model, fr_param, pm.summary(trace_fr), raw_mapping, name_mapping)

      # Saving raw ADVI results
      
      mf_df = pd.DataFrame(mf_param)
      fr_df = pd.DataFrame(fr_param)
      
      # Loading persisted trace
   
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/', model=airline_model)
      
      # Traceplots
      
      pa.traceplots(trace_hmc, varnames, ml_deltas, 3, combined=False)
      pa.traceplots(trace_mf, varnames, ml_deltas, 3, True)
      pa.traceplots(trace_fr, varnames, ml_deltas, 5, True)
      
      # Traceplots compare
      
      pa.traceplot_compare(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, ml_deltas, rv_mapping, 6)
      plt.suptitle('Marginal Hyperparameter Posteriors', fontsize='small')
     
      # Prior Posterior Plot
      
      pa.plot_prior_posterior_plots(trace_prior, trace_hmc, varnames, ml_deltas, 'Prior Posterior HMC')
      pa.plot_prior_posterior_plots(trace_prior, trace_mf, varnames, ml_deltas, 'Prior Posterior MF')
      pa.plot_prior_posterior_plots(trace_prior, trace_fr, varnames, ml_deltas, 'Prior Posterior FR')
      
      pa.traceplots_two_way_compare(trace_mf, trace_fr, varnames, ml_deltas, 'Posteriors MF / FR', 'MF', 'FR')

      # Autocorrelations
      
      pm.autocorrplot(trace_hmc, varnames)
      
      # Saving summary stats 
      
      hmc_summary_df = pm.summary(trace_hmc).ix[varnames]
      hmc_summary_df.to_csv(results_path + '/hmc_summary_df.csv', sep=',')

      # Pair Grid plots 
      
      trace = trace_mf
      clr='coral'
      
      k1_names = ['s_1', 'ls_2', 'ls_3']
      k2_names = ['s_4', 'ls_5', 'n_6']
      
      trace_k1 = pa.get_subset_trace(trace, k1_names)
      trace_k2 = pa.get_subset_trace(trace, k2_names)
      
      pa.pair_grid_plot(trace_k1, ml_deltas, k1_names, color=clr)
      pa.pair_grid_plot(trace_k2, ml_deltas, k2_names, color=clr)
      
      
      # Pair scatter plot 
      from itertools import combinations

      bi_list = []
      for i in combinations(varnames, 2):
            bi_list.append(i)
            
      
      for i, j  in zip(bi_list, np.arange(len(bi_list))):
        print(i)
        print(j)
        if np.mod(j,8) == 0:
            fig = plt.figure(figsize=(15,8))
        plt.subplot(2,4,np.mod(j, 8)+1)
        sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=0.9)
        #sns.kdeplot(trace_hmc[i[0]], trace_hmc[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        #sns.scatterplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', size=1, legend=False)
        #sns.scatterplot(trace_fr[i[0]], trace_fr[i[1]], color='g', size=1, legend=False)
        plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
      
      
      # Testing convergence of ADVI - TODO 
      
      ad.convergence_report(tracker_mf, mf_param, varnames,  mf.hist, 'Mean Field Convergence Report')
      pa.convergence_report(tracker_fr, fr_param, varnames, fr.hist, 'Full Rank Convergence Report')
            
      # Predictions

      # HMC
      
      pa.write_posterior_predictive_samples(trace_hmc, 10, t_test, results_path + 'pred_dist/', method='hmc_final', gp=gp) 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc_final.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc_final.csv')
      
      sample_means_hmc = forward_mu(sample_means_hmc, emp_mu, emp_std)
      sample_stds_hmc = forward_std(sample_stds_hmc, emp_std)
      
      mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
      
      rmse_hmc = pa.rmse(mu_hmc, forward_mu(y_test, emp_mu, emp_std))
      lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_hmc, sample_stds_hmc)
      
      
      plt.figure()
      plt.plot(t_test, sample_means_hmc.T, 'grey', alpha=0.2)
      plt.plot(t_test, mu_test, 'r-')
      plt.plot(t_test, forward_mu(y_test, emp_mu, emp_std), 'ko')
      plt.plot(t_test, mu_hmc, 'b-')
      plt.fill_between(t_test.flatten(), lower_hmc, upper_hmc, color='blue', alpha=0.2)
      plt.fill_between(t_test.flatten(), mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)

     
      # MF
      
      pa.write_posterior_predictive_samples(trace_mf, 20, t_test, results_path + 'pred_dist/', method='mf', gp=gp) 
      
      sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
      sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')
      
      sample_means_mf = forward_mu(sample_means_mf, emp_mu, emp_std)
      sample_stds_mf = forward_std(sample_stds_mf, emp_std)
      
      mu_mf = pa.get_posterior_predictive_mean(sample_means_mf)
      lower_mf, upper_mf = pa.get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)
      
      rmse_mf = pa.rmse(mu_mf, forward_mu(y_test, emp_mu, emp_std))
      lppd_mf, lpd_mf = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_mf, sample_stds_mf)


      # FR
      
      pa.write_posterior_predictive_samples(trace_fr, 20, t_test, results_path +  'pred_dist/', method='fr', gp=gp) 
      
      sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr.csv')
      sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')
      
      sample_means_fr = forward_mu(sample_means_fr, emp_mu, emp_std)
      sample_stds_fr = forward_std(sample_stds_fr, emp_std)
      
      mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
      lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
      rmse_fr = pa.rmse(mu_fr, forward_mu(y_test, emp_mu, emp_std))
      lppd_fr, lpd_fr = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_fr, sample_stds_fr)
                  
      
      # Plot HMC vs ML vs MF vs FR / All-in 

      plt.figure(figsize=(18,10))
      plt.subplot(241)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      #plt.plot(df['Year'][0:sep_idx], mu_fit, alpha=0.5, label='train', color='k')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, label='test', color='r')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'NLPD: ' + str(-lpd_), fontsize='small')
      
      plt.subplot(242)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.5, label='test', color='b')
      #plt.plot(df['Year'][sep_idx:], sample_means_hmc.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('HMC' + '\n' + 'RMSE: ' + str(rmse_hmc) + '\n' + 'NLPD: ' + str(-lpd_hmc), fontsize='small')
      
                  
      plt.subplot(243)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, label='test', color='coral')
      #plt.plot(df['Year'][sep_idx:], sample_means_hmc.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('MF' + '\n' + 'RMSE: ' + str(rmse_mf) + '\n' + 'NLPD: ' + str(-lpd_mf), fontsize='small')
      
      plt.subplot(244)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.5, label='test', color='g')
      #plt.plot(df['Year'][sep_idx:], sample_means_fr.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('FR' + '\n' + 'RMSE: ' + str(rmse_fr) + '\n' + 'NLPD: ' + str(-lpd_fr), fontsize='small')
      
      
      # All in 3-way
      
      plt.subplot(245)
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.4, label='HMC')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2, label='ML-II')
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, color='r')
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.5, color='b')
      plt.legend(fontsize='small')
      plt.title('ML-II vs HMC', fontsize='small')

      
      plt.subplot(246)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.4, label='HMC')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.4, label='FR')
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.8, color='b')
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.8, color='g')
      plt.legend(fontsize='small')
      plt.title('HMC vs FR',fontsize='small')
      
      
      plt.subplot(247)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.4, label='MF')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.4, label='FR')
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, color='coral')
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.5, color='g')
      plt.legend(fontsize='small')
      plt.title('MF vs FR',fontsize='small')
      
      
      plt.subplot(248)      
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.4, label='MF')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2, label='ML-II')
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, color='coral')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, color='r')
      plt.legend(fontsize='small')
      plt.title('ML-II vs MF',fontsize='small')