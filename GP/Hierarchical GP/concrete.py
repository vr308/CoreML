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


def multid_traceplot(trace_hmc_df, varnames, feature_mapping, ml_deltas_log):
      
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7], [0,1,2,3]):
            plt.subplot(4,2,i)
            plt.hist(trace_hmc_df[varnames[j]], bins=100, alpha=0.4)
            plt.axvline(x=ml_deltas_log[varnames[j]], color='r')
            plt.title(varnames[j] + ' / ' + feature_mapping[varnames[j]], fontsize='small')
            plt.subplot(4,2,i+1)
            plt.plot(trace_hmc_df[varnames[j]], alpha=0.4)
      plt.tight_layout()
            
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7], [4,5,6,7]):
            plt.subplot(4,2,i)
            plt.hist(trace_hmc_df[varnames[j]], bins=100, alpha=0.4)
            plt.axvline(x=ml_deltas_log[varnames[j]], color='r')
            plt.title(varnames[j] + ' / ' + feature_mapping[varnames[j]], fontsize='small')
            plt.subplot(4,2,i+1)
            plt.plot(trace_hmc_df[varnames[j]], alpha=0.4)
      plt.tight_layout()
      
      
if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Concrete/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = uni_path
      
      raw = pd.read_csv(path + 'concrete.csv', keep_default_na=False)
      
      #df = normalize(raw)
      
      df = np.array(raw)
      
      str_names = ['cement','slag','ash','water','superplasticize','coarse_aggregate','fine_aggregate','age']
      
      mu_y = np.mean(raw['strength'])
      std_y = np.std(raw['strength'])
      
     # y = df['strength']
     # X = df[['cement','slag','ash','water','superplasticize','coarse_aggregate','fine_aggregate','age']]
      
      y = df[:,-1]
      X = df[:,0:8]
      
      n = len(X)
      
      dim_train =  515 #50% of the data
      
      train_id = np.random.choice(np.arange(n), size=dim_train, replace=False)
      test_id = ~np.isin(np.arange(n), train_id)
      
      y_train = y[train_id]
      y_test = y[test_id]
      
      X_train = X[train_id]
      X_test = X[test_id]
      
      def forward_mu(test, mu_y, std_y):
            
            return test*std_y + mu_y
                 
      def forward_std(test, std_y):
            
            return test*std_y
      
      # ML-II 
      
      # sklearn kernel 
      
      # se-ard + noise
      
      se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*8), length_scale_bounds=(0.000001,1e5))
     
      noise = WhiteKernel(noise_level=1**2,
                        noise_level_bounds=(1e-5, 100))  # noise terms
      
      sk_kernel = se_ard + noise
      
#      models = []
#      
#      for i in [0,1,2]:
#      
#            gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=30)
#            models.append(gpr.fit(X_train, y_train))
#            
      gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=10)
      gpr.fit(X_train, y_train)
       
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
#      rmse_agg = []; lpd_agg = [];
#      
#      for i in np.arange(3):
#            
#            gpr = models[i]
#      
#            #mu_fit, std_fit = gpr.predict(X_train, return_std=True)      
#            mu_test, std_test = gpr.predict(X_test, return_std=True)
#                  
#            rmse_agg.append(pa.rmse(mu_test, y_test))
#            
#            lpd_agg.append(pa.log_predictive_density(y_test, mu_test, std_test))
#            
      # No plotting 
      
      gpr = models[2]
      
      mu_test, std_test = gpr.predict(X_test, return_std=True)
      
      rmse_ = pa.rmse(mu_test, y_test)
      se_rmse = pa.se_of_rmse(mu_test, y_test)
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      
      
      # Linear regression to double check with Hugh DSVI paper / sanity check
      
      from sklearn import linear_model
      
      regr = linear_model.LinearRegression()
      
      sep_idx = 515
      regr.fit(raw[str_names][0:sep_idx], raw['strength'][0:sep_idx])
      
      y_pred = regr.predict(raw[str_names][sep_idx:])
      
      rmse_lin = pa.rmse(y_pred, raw['strength'][sep_idx:])
      lpd_lin = pa.log_predictive_density(y_pred, raw['strength'][sep_idx:])
      se_rmse_lin = pa.se_of_rmse(y_pred, raw['strength'][sep_idx:])
                  
      # Get prediction interval 
      
      lower = mu_test - 1.96*std_test
      upper = mu_test + 1.96*std_test
      
      s = np.sqrt(gpr.kernel_.k1.k1.constant_value)
      ls =  gpr.kernel_.k1.k2.length_scale
      n = np.sqrt(gpr.kernel_.k2.noise_level)
      
      ml_deltas = {'s':s,'ls':ls, 'n': n}
      varnames = ['s', 'ls','n']
      
      feature_mapping = {'log_s': '', 
                 'log_n': '', 
                 'log_ls__0': str_names[0],
                 'log_ls__1': str_names[1],
                 'log_ls__2': str_names[2],
                 'log_ls__3': str_names[3],
                 'log_ls__4': str_names[4],
                 'log_ls__5': str_names[5],
                 'log_ls__6': str_names[6],
                 'log_ls__7': str_names[7]
                 }
      
      feature_mapping2 = {'s': '', 
                 'n': '', 
                 'ls__0': str_names[0],
                 'ls__1': str_names[1],
                 'ls__2': str_names[2],
                 'ls__3': str_names[3],
                 'ls__4': str_names[4],
                 'ls__5': str_names[5],
                 'ls__6': str_names[6],
                 'ls__7': str_names[7]
                 }

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
      
      varnames_unravel = np.array(list(ml_deltas_unravel.keys()))
      varnames_log_unravel = np.array(list(ml_deltas_log.keys()))
      
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
           
           log_s = pm.Normal('log_s', 0, 3)
           log_ls = pm.Normal('log_ls', mu=np.array([0]*n_dim), sd=np.ones(n_dim,)*3, shape=(n_dim,))
           log_n = pm.Normal('log_n', 0, 3)
           
           s = pm.Deterministic('s', tt.exp(log_s))
           ls = pm.Deterministic('ls', tt.exp(log_ls))
           n = pm.Deterministic('n', tt.exp(log_n))

           # Specify the covariance function
       
           cov_main = pm.gp.cov.Constant(s**2)*pm.gp.cov.ExpQuad(n_dim, ls)
           cov_noise = pm.gp.cov.WhiteNoise(n**2)
       
           gp_main = pm.gp.Marginal(cov_func=cov_main)
           gp_noise = pm.gp.Marginal(cov_func=cov_noise) 
           
           gp = gp_main
           
           trace_prior = pm.sample(500)
           
      with concrete_model:
            
           # Marginal Likelihood
           y_ = gp.marginal_likelihood("y", X=X_train, y=y_train, noise=cov_noise)
       
      with concrete_model:
      
            trace_hmc = pm.sample(draws=500, tune=500, chains=2)
               
      with concrete_model:
    
            pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc/', overwrite=True)
      
      with concrete_model:
      
            trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/')
        
      with concrete_model:
            
            mf = pm.ADVI()
      
            tracker_mf = pm.callbacks.Tracker(
            mean = mf.approx.mean.eval,    
            std = mf.approx.std.eval)
           
            mf.fit(n=50000, callbacks=[tracker_mf])
            
            trace_mf = mf.approx.sample(4000)
      
      with concrete_model:
            
            fr = pm.FullRankADVI()
              
            tracker_fr = pm.callbacks.Tracker(
            mean = fr.approx.mean.eval,    
            std = fr.approx.std.eval)
            
            fr.fit(n=50000, callbacks=[tracker_fr])
            trace_fr = fr.approx.sample(4000)
            
            
            bij_mf = mf.approx.groups[0].bij 
            mf_param = {param.name: bij_mf.rmap(param.eval()) for param in mf.approx.params}
      
            bij_fr = fr.approx.groups[0].bij
            fr_param = {param.name: bij_fr.rmap(param.eval()) for param in fr.approx.params}

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
   
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/', model=concrete_model)
      
      # Traceplots
      
      pa.traceplots(trace_hmc, ['s', 'n'], ml_deltas, 2, combined=True, clr='b')
      pa.traceplots(trace_mf, ['s', 'n'], ml_deltas, 2, combined=True,clr='coral')
      pa.traceplots(trace_fr, ['s','n'], ml_deltas, 2, combined=True, 'g')
      
      multid_traceplot(trace_hmc_df, varnames_unravel, feature_mapping2, ml_deltas_unravel)

      # Forest plot
      
      pm.forestplot(trace_hmc, varnames=varnames, rhat=True, quartiles=False)
      pm.forestplot(trace_mf, varnames=varnames, rhat=True, quartiles=False)
      plt.title('95% Credible Intervals (Mean-Field VI)', fontsize='small')
      
      pm.forestplot(trace_fr, varnames=varnames, rhat=True, quartiles=False)
      plt.title('95% Credible Intervals (Full-Rank VI)', fontsize='small')

      # Convergence 
   
      ad.convergence_report(tracker_mf, mf.hist, varnames_log_unravel, 'Mean-Field Convergence')
      ad.convergence_report(tracker_fr, fr.hist, varnames_log_unravel, 'Full-Rank Convergence')
      
      # Traceplots compare
      
      pa.traceplot_compare(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, ml_deltas, rv_mapping, 6)
      plt.suptitle('Marginal Hyperparameter Posteriors', fontsize='small')
     
      # Prior Posterior Plot
      
      pa.plot_prior_posterior_plots(trace_prior_df, trace_hmc_df, varnames_log_unravel, ml_deltas_log, 'Prior Posterior HMC')
      pa.plot_prior_posterior_plots(trace_prior_df, trace_mf_df, varnames_log_unravel, ml_deltas_log, 'Prior Posterior MF')
      pa.plot_prior_posterior_plots(trace_prior_df, trace_fr_df, varnames_log_unravel, ml_deltas_log, 'Prior Posterior FR')
      
      pa.traceplots_two_way_compare(trace_mf, trace_fr, varnames, ml_deltas, 'Posteriors MF / FR', 'MF', 'FR')

      # Autocorrelations
      
      pm.autocorrplot(trace_hmc, varnames)
      
      # Saving summary stats 
      
      hmc_summary_df = pm.summary(trace_hmc)
      hmc_summary_df.to_csv(results_path + '/hmc_summary_df.csv', sep=',')

      # Pair Grid plots 
      
      clr='b'
      
      varnames_unravel = ['s', 'ls__0', 'ls__1', 'ls__2','ls__3','ls__4','ls__7', 'n']
            
      #pa.pair_grid_plot(trace_hmc_df[varnames_log], ml_deltas_log, varnames_log, color=clr)      
      
      # Pair scatter plot 
      from itertools import combinations

      bi_list = []
      for i,j in zip(combinations(varnames_unravel, 2)):
            print(i)
            bi_list.append(i)
            
      
      for i, j  in zip(bi_list, np.arange(len(bi_list))):
        print(i)
        print(j)
        if np.mod(j,8) == 0:
            fig = plt.figure(figsize=(15,8))
        plt.subplot(2,4,np.mod(j, 8)+1)
        #sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=0.9)
        sns.kdeplot(trace_hmc_df[i[0]], trace_hmc_df[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        #sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        #sns.scatterplot(trace_hmc_df[i[0]], trace_hmc_df[i[1]], color='b', size=0.5, legend=False)
        #sns.scatterplot(trace_fr[i[0]], trace_fr[i[1]], color='g', size=1, legend=False)
        plt.scatter(ml_deltas_unravel[i[0]], ml_deltas_unravel[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
      
       
      # Predictions

      # HMC
      
      pa.write_posterior_predictive_samples(trace_hmc, 10, X_test, results_path + 'pred_dist/', method='hmc', gp=gp) 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc.csv')
      
      #sample_means_hmc = forward_mu(sample_means_hmc, emp_mu, emp_std)
      #sample_stds_hmc = forward_std(sample_stds_hmc, emp_std)
      
      mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
      
      rmse_hmc = pa.rmse(mu_hmc, y_test)
      se_rmse_hmc = pa.se_of_rmse(mu_hmc, y_test)
      lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
      
      # MF
      
      pa.write_posterior_predictive_samples(trace_mf, 20, t_test, results_path + 'pred_dist/', method='mf', gp=gp) 
      
      sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
      sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')
      
      #sample_means_mf = forward_mu(sample_means_mf, emp_mu, emp_std)
      #sample_stds_mf = forward_std(sample_stds_mf, emp_std)
      
      mu_mf = pa.get_posterior_predictive_mean(sample_means_mf)
      lower_mf, upper_mf = pa.get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)
      
      rmse_mf = pa.rmse(mu_mf, forward_mu(y_test, emp_mu, emp_std))
      se_rmse_mf = pa.se_of_rmse(mu_mf, y_test)
      lppd_mf, lpd_mf = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_mf, sample_stds_mf)


      # FR
      
      pa.write_posterior_predictive_samples(trace_fr, 20, t_test, results_path +  'pred_dist/', method='fr', gp=gp) 
      
      sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr.csv')
      sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')
      
      #sample_means_fr = forward_mu(sample_means_fr, emp_mu, emp_std)
      #sample_stds_fr = forward_std(sample_stds_fr, emp_std)
      
      mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
      lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
      rmse_fr = pa.rmse(mu_fr, forward_mu(y_test, emp_mu, emp_std))
      se_rmse_fr = pa.se_of_rmse(mu_fr, y_test)
      lppd_fr, lpd_fr = pa.log_predictive_mixture_density(forward_mu(y_test, emp_mu, emp_std), sample_means_fr, sample_stds_fr)
      
      
      
      
      
      
      
      
      
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       