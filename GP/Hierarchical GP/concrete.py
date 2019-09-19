#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:40:26 2019

@author: vidhi
"""

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pylab as plt
import  scipy.stats as st 
import seaborn as sns
from sklearn.preprocessing import normalize
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa
import advi_analysis as ad


def multid_traceplot(trace_hmc, str_names):
      
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7], [0,1,2,3]):
            plt.subplot(4,2,i)
            plt.hist(trace_hmc['ls'][:,j], bins=100, alpha=0.4)
            plt.xscale('log')
            plt.axvline(x=ml_deltas['ls'][j], color='r')
            plt.title('ls_' + str(j) + ' / ' + str_names[j], fontsize='small')
            plt.subplot(4,2,i+1)
            plt.plot(trace_hmc['ls'][:,j], alpha=0.4)
            
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7], [4,5,6,7]):
            plt.subplot(4,2,i)
            plt.hist(trace_hmc['ls'][:,j], bins=100, alpha=0.4)
            plt.xscale('log')
            plt.axvline(x=ml_deltas['ls'][j], color='r')
            plt.title('ls_' + str(j) + ' / ' + str_names[j], fontsize='small')
            plt.subplot(4,2,i+1)
            plt.plot(trace_hmc['ls'][:,j], alpha=0.4)
      
      
if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Concrete/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = uni_path
      
      raw = pd.read_csv(path + 'concrete.csv', keep_default_na=False)
      
      df = normalize(raw)
      
      str_names = ['cement','slag','ash','water','superplasticize','coarse_aggregate','fine_aggregate','age']
      
     # y = df['strength']
     # X = df[['cement','slag','ash','water','superplasticize','coarse_aggregate','fine_aggregate','age']]
      
      y = df[:,-1]
      X = df[:,0:8]
      
      sep_idx = 500
      
      y_train = y[0:sep_idx]
      y_test = y[sep_idx:]
      
      X_train = X[0:sep_idx]
      X_test = X[sep_idx:]
      
      def forward_mu():
            
            
            
      def forward_std():
      # ML-II 
      
      # sklearn kernel 
      
      # se-ard + noise
      
      se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*8), length_scale_bounds=(0.000001,1e5))
     
      noise = WhiteKernel(noise_level=1**2,
                        noise_level_bounds=(1e-2, 100))  # noise terms
      
      sk_kernel = se_ard + noise
      
      models = []
      
      for i in [0,1,2]:
      
            gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=20)
            models.append(gpr.fit(X_train, y_train))
       
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
      for i in np.arange(6):
            
            gpr = models[i]
      
            mu_fit, std_fit = gpr.predict(X_train, return_std=True)      
            mu_test, std_test = gpr.predict(X_test, return_std=True)
                  
            rmse_ = pa.rmse(mu_test, y_test)
            print(rmse_)
            
            lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
            print(lpd_)
            
      # No plotting 
      
      s = np.sqrt(gpr.kernel_.k1.k1.constant_value)
      ls =  gpr.kernel_.k1.k2.length_scale
      n = np.sqrt(gpr.kernel_.k2.noise_level)
      
      ml_deltas = {'s':s,'ls':ls, 'n': n}
      varnames = ['s', 'ls','n']
      
      ml_deltas_unravel = {'s':s,
                           'ls__0':ls[0],
                           'ls__1':ls[1],
                           'ls__2':ls[2],
                           'ls__3':ls[3],
                           'ls__4':ls[4],
                           'ls__5':ls[5],
                           'ls__6':ls[6],
                           'ls__7':ls[7],
                           'n': n}
      
      ml_deltas_log = { 'log_n': np.log(n), 
                       'log_ls__0': np.log(ls[0]),
                       'log_ls__1': np.log(ls[1]),
                       'log_ls__2': np.log(ls[2]),
                       'log_ls__3': np.log(ls[3]),
                       'log_ls__4': np.log(ls[4]),
                       'log_ls__5': np.log(ls[5]),
                       'log_ls__6': np.log(ls[6]),
                       'log_ls__7': np.log(ls[7]),
                       'log_s': np.log(s),
                       }
      
      
      # Plotting ML-II high-d
      
#      plt.figure(figsize=(16,9))
#      plt.subplot(241)
#      plt.plot(raw['cement'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      plt.plot(raw['cement'][0:sep_idx], mu_fit*std_y + mu_y, 'ro', markersize=1)
#      
#      plt.subplot(242)
#      plt.plot(raw['slag'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      plt.plot(raw['slag'][0:sep_idx], mu_fit*std_y + mu_y, 'ro', markersize=1)
#
#      
#      plt.subplot(243)
#      plt.plot(raw['ash'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      plt.plot(raw['ash'][0:sep_idx], mu_fit*std_y + mu_y, 'ro', markersize=1)
#
#      plt.subplot(244)
#      plt.plot(raw['water'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      
#      plt.subplot(245)
#      plt.plot(raw['superplasticize'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      
#      plt.subplot(246)
#      plt.plot(raw['coarse_aggregate'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      
#      plt.subplot(247)
#      plt.plot(raw['fine_aggregate'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      
#      plt.subplot(248)
#      plt.plot(raw['age'][0:sep_idx], raw['strength'][0:sep_idx], 'kx', markersize=1)
#      
     
     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------
     
      with pm.Model() as concrete_model:
           
           log_s = pm.Normal('log_s', 0, 2)
           log_ls = pm.Normal('log_ls', mu=np.array([0]*8), sd=np.ones(8,)*2, shape=(8,))
           log_n = pm.Normal('log_n', 0, 2)
           
           s = pm.Deterministic('s', tt.exp(log_s))
           ls = pm.Deterministic('ls', tt.exp(log_ls))
           n = pm.Deterministic('n', tt.exp(log_n))
           
          
           # Specify the covariance function
       
           cov_main = pm.gp.cov.Constant(s**2)*pm.gp.cov.ExpQuad(8, ls)
           cov_noise = pm.gp.cov.WhiteNoise(n**2)
       
           gp_main = pm.gp.Marginal(cov_func=cov_main)
           gp_noise = pm.gp.Marginal(cov_func=cov_noise) 
           
           gp = gp_main
           
           trace_prior = pm.sample(500)
           
      with concrete_model:
            
           # Marginal Likelihood
           y_ = gp.marginal_likelihood("y", X=X_train, y=y_train, noise=cov_noise)
       
      with concrete_model:
      
            trace_hmc = pm.sample(draws=1000, tune=500, chains=1)
               
      with concrete_model:
    
            pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc/', overwrite=True)
      
      with concrete_model:
      
            trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/')
        
      with concrete_model:
            
            mf = pm.ADVI()
      
            tracker_mf = pm.callbacks.Tracker(
            mean = mf.approx.mean.eval,    
            std = mf.approx.std.eval)
           
            mf.fit(n=80000, callbacks=[tracker_mf])
            
            trace_mf = mf.approx.sample(4000)
      
      with concrete_model:
            
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
      
      mf_param = ad.analytical_variational_opt(concrete_model, mf_param, pm.summary(trace_mf), raw_mapping, name_mapping)
      fr_param = ad.analytical_variational_opt(concrete_model, fr_param, pm.summary(trace_fr), raw_mapping, name_mapping)

      # Saving raw ADVI results
      
      mf_df = pd.DataFrame(mf_param)
      fr_df = pd.DataFrame(fr_param)
      
      # Converting trace to df
      
      trace_prior_df = pm.trace_to_dataframe(trace_prior)
      trace_hmc_df = pm.trace_to_dataframe(trace_hmc)
      trace_mf_df = pm.trace_to_dataframe(trace_mf)
      trace_fr_df = pm.trace_to_dataframe(trace_fr)
      
      # Loading persisted trace
   
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/', model=airline_model)
      
      # Traceplots
      
      pa.traceplots(trace_hmc, ['s', 'n'], ml_deltas, 2, combined=False, clr='b')
      pa.traceplots(trace_mf, ['s', 'n'], ml_deltas, 3, combined=True,clr='coral')
      pa.traceplots(trace_fr, ['s','n'], ml_deltas, 2, combined=True, 'g')
      
      # Convergence 
      
      varnames_log = ['log_n', 'log_ls__0', 'log_ls__1', 'log_ls__2', 'log_ls__3',
       'log_ls__4', 'log_ls__5', 'log_ls__6', 'log_ls__7', 'log_s']
      
      ad.convergence_report(tracker_mf, varnames_log, 'Mean-Field Convergence')
      ad.convergence_report(tracker_fr, varnames_log, 'Full-Rank Convergence')
      
      # Traceplots compare
      
      pa.traceplot_compare(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, ml_deltas, rv_mapping, 6)
      plt.suptitle('Marginal Hyperparameter Posteriors', fontsize='small')
     
      # Prior Posterior Plot
      
      varnames_log = ['log_n', 'log_ls__0', 'log_ls__1', 'log_ls__2', 'log_ls__3',
       'log_ls__4', 'log_ls__5', 'log_ls__6', 'log_ls__7', 'log_s']
            
      pa.plot_prior_posterior_plots(trace_prior_df, trace_hmc_df, varnames_log, ml_deltas_log, 'Prior Posterior HMC')
      pa.plot_prior_posterior_plots(trace_prior_df, trace_mf_df, varnames_log, ml_deltas_log, 'Prior Posterior MF')
      pa.plot_prior_posterior_plots(trace_prior_df, trace_fr_df, varnames_log, ml_deltas_log, 'Prior Posterior FR')
      
      pa.traceplots_two_way_compare(trace_mf, trace_fr, varnames, ml_deltas, 'Posteriors MF / FR', 'MF', 'FR')

      # Autocorrelations
      
      pm.autocorrplot(trace_hmc, varnames)
      
      # Saving summary stats 
      
      hmc_summary_df = pm.summary(trace_hmc)
      hmc_summary_df.to_csv(results_path + '/hmc_summary_df.csv', sep=',')

      # Pair Grid plots 
      
      clr='b'
      
      varnames_unravel = ['s', 'ls__0', 'ls__1', 'ls__2','ls__3','ls__4','ls__5','ls__6','ls__7', 'n']
            
      pa.pair_grid_plot(trace_hmc_df[varnames_log], ml_deltas_log, varnames_log, color=clr)      
      
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
      
       
      # Predictions

      # HMC
      
      pa.write_posterior_predictive_samples(trace_hmc, 10, X_test, results_path + 'pred_dist/', method='hmc', gp=gp) 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc.csv')
      
      sample_means_hmc = forward_mu(sample_means_hmc, emp_mu, emp_std)
      sample_stds_hmc = forward_std(sample_stds_hmc, emp_std)
      
      mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
      
      rmse_hmc = pa.rmse(mu_hmc, y_test)
      lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_hmc, sample_stds_hmc)
      
      # MF
      
        pa.write_posterior_predictive_samples(trace_mf, 20, t_test, results_path + 'pred_dist/', method='mf', gp=gp) 
      
      sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
      sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')
      
      sample_means_mf = forward_mu(sample_means_mf, emp_mu, emp_std)
      sample_stds_mf = forward_std(sample_stds_mf, emp_std)
      
      mu_mf = pa.get_posterior_predictive_mean(sample_means_mf)
      #lower_mf, upper_mf = pa.get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)
      
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
      
      
      
      
      
      
      
      
      
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       