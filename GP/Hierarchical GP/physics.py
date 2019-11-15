#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:18:41 2019

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import posterior_analysis as pa
import advi_analysis as ad

def multid_traceplot(trace_hmc_df, varnames, feature_mapping, ml_deltas_log):
      
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7, 9, 11,13], [0,1,2,3,4,5,6]):
            plt.subplot(7,2,i)
            plt.hist(trace_hmc_df[varnames[j]], bins=100, alpha=0.4)
            plt.axvline(x=ml_deltas_log[varnames[j]], color='r')
            plt.title(varnames[j] + ' / ' + feature_mapping[varnames[j]], fontsize='small')
            plt.subplot(7,2,i+1)
            plt.plot(trace_hmc_df[varnames[j]], alpha=0.4)
      plt.tight_layout()
            
      plt.figure(figsize=(16,9))
      
      for i,j in zip([1,3,5,7, 9, 11], [7,8,9,10,11,12]):
            plt.subplot(6,2,i)
            plt.hist(trace_hmc_df[varnames[j]], bins=100, alpha=0.4)
            plt.axvline(x=ml_deltas_log[varnames[j]], color='r')
            plt.title(varnames[j] + ' / ' + feature_mapping[varnames[j]], fontsize='small')
            plt.subplot(6,2,i+1)
            plt.plot(trace_hmc_df[varnames[j]], alpha=0.4)
      plt.tight_layout()
      
      
if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Physics/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Physics/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Physics/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Physics/'

      path = uni_path
      
      raw = pd.read_csv(path + 'physics.csv', keep_default_na=False)
      
      #mu_y = np.mean(raw['objective'])
      #std_y = np.std(raw['objective'])
      
      #df = normalize(raw)
      
      df = np.array(raw)
      
      y = df[:,-1]
      X = df[:,0:13]
      
      df = -df
      sub_df = df[df[:,-1] < 0]
      
      X = df[:,[0,2,3,4,6,8,11]]

      str_names = list(raw.columns[0:13])
      
      n_all = len(X)
      n_dim = len(str_names)
      
      n_train = 280 #50% of the data
      
      train_id = np.random.choice(np.arange(n_all), size=n_train, replace=False)
      
      train_id.tofile(results_path + 'train_id.csv', sep=',')
      
      train_id = np.array(pd.read_csv(results_path + 'train_id.csv', header=None)).reshape(n_train,)
      
      test_id = ~np.isin(np.arange(n_all), train_id)
      
      y_train = y[train_id]
      y_test = y[test_id]
      
      #X_train = X[train_id]
      #X_test = X[test_id]
      
      X_train = X
      y_train = y
      
#      train_id = []; test_id = []
#      
#      kf = KFold(n_splits=5, shuffle=False)
#      
#      X_train_folds = []; X_test_folds = []; y_train_folds = []; y_test_folds = []
#      
#      for i, j in kf.split(X):
#          #print(i)
#          #print(j)
#          X_train_folds.append(X[i])
#          X_test_folds.append(X[j])
#          y_train_folds.append(y[i])
#          y_test_folds.append(y[j])
      
      #sep_idx = 200
      #y_train = y[0:sep_idx]
      #y_test = y[sep_idx:]
      
      #X_train = X[0:sep_idx]
      #X_test = X[sep_idx:]
      
      
      # ML-II 
      
      # sklearn kernel 
      
      # se-ard + noise
      
      se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*13), length_scale_bounds=(0.000001,1e5))
     
      noise = WhiteKernel(noise_level=1**2,
                        noise_level_bounds=(1e-5, 100))  # noise terms
      
      sk_kernel = se_ard + noise
      
      gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=5)
      gpr.fit(X_train, y_train)
      
#      models = []
#      lml = []
#      
#      for i in np.arange(len(X_train_folds)):
#      
#            gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=10)
#            models.append(gpr.fit(X_train_folds[i], y_train_folds[i]))
#            lml.append(gpr.log_marginal_likelihood_value_)
       
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
            
#      mu_test_agg = []
#      std_test_agg = []
#      rmse_agg = []
#      se_rmse_agg = []
#      lpd_agg = []
#      
#      #mu_fit, std_fit = gpr.predict(X_train, return_std=True) 
#      for i in np.arange(len(models)):
#            mu_test, std_test = models[i].predict(X_test_folds[i], return_std=True)
#            mu_test_agg.append(mu_test)
#            std_test_agg.append(std_test)
#            
#            rmse_agg.append(pa.rmse(mu_test, y_test_folds[i]))
#            se_rmse_agg.append(pa.se_of_rmse(mu_test, y_test_folds[i]))
#            lpd_agg.append(pa.log_predictive_density(y_test_folds[i], mu_test, std_test))
#            
#      #mu_test, std_test = gpr.predict(X_test, return_std=True)
#      
#      gpr = models[0]
            
      mu_test, std_test = gpr.predict(X_test, return_std=True)
      
      rmse_ = pa.rmse(mu_test, y_test)
      se_rmse = pa.se_of_rmse(mu_test, y_test)
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      
      print('rmse_ml: ' + str(rmse_))
      print('se_rmse_ml: ' + str(se_rmse))
      print('lpd_:' + str(lpd_))
      
       # Write down mu_ml

      np.savetxt(fname=results_path + 'pred_dist/' + 'means_ml.csv', X=mu_test, delimiter=',', header='')
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
                           'ls__8':ls[8],
                           'ls__9':ls[9],
                           'ls__10':ls[10],
                           'ls__11':ls[11],
                           'ls__12':ls[12],
                           'n': n}
      
      ml_deltas_log = {'log_s': np.log(s), 
                       'log_n': np.log(n), 
                       'log_ls__0': np.log(ls[0]),
                       'log_ls__1': np.log(ls[1]),
                       'log_ls__2': np.log(ls[2]),
                       'log_ls__3': np.log(ls[3]),
                       'log_ls__4': np.log(ls[4]),
                       'log_ls__5': np.log(ls[5]),
                       'log_ls__6': np.log(ls[6]),
                       'log_ls__7': np.log(ls[7]),
                       'log_ls__8': np.log(ls[8]),
                       'log_ls__9': np.log(ls[9]),
                       'log_ls__10': np.log(ls[10]),
                       'log_ls__11': np.log(ls[11]),
                       'log_ls__12': np.log(ls[12])
                       }
      
      ml_deltas_log = {'log_s': np.log(s), 
                       'log_n': np.log(n), 
                       'log_ls__0': np.log(ls[0]),
                       #'log_ls__1': np.log(ls[1]),
                       'log_ls__2': np.log(ls[1]),
                       'log_ls__3': np.log(ls[2]),
                       'log_ls__4': np.log(ls[3]),
                       #'log_ls__5': np.log(ls[5]),
                       'log_ls__6': np.log(ls[4]),
                       #'log_ls__7': np.log(ls[7]),
                       'log_ls__8': np.log(ls[5]),
                       #'log_ls__9': np.log(ls[9]),
                       #'log_ls__10': np.log(ls[10]),
                       'log_ls__11': np.log(ls[5]),
                       #'log_ls__12': np.log(ls[12])
                       }
      
      feature_mapping = {'log_s': '', 
                       'log_n': '', 
                       'log_ls__0': str_names[0],
                       'log_ls__1': str_names[1],
                       'log_ls__2': str_names[2],
                       'log_ls__3': str_names[3],
                       'log_ls__4': str_names[4],
                       'log_ls__5': str_names[5],
                       'log_ls__6': str_names[6],
                       'log_ls__7': str_names[7],
                       'log_ls__8': str_names[8],
                       'log_ls__9': str_names[9],
                       'log_ls__10': str_names[10],
                       'log_ls__11': str_names[11],
                       'log_ls__12': str_names[12]
                       }
      
      varnames_unravel = np.array(list(ml_deltas_unravel.keys()))
      varnames_log_unravel = np.array(list(ml_deltas_log.keys()))
      
      with pm.Model() as physics_model:
           
           log_s = pm.Normal('log_s', 0, 3)
           #log_ls = pm.Normal('log_ls', mu=np.array([0]*13), sd=np.ones(13,)*3, shape=(13,))
           log_n = pm.Normal('log_n', ml_deltas['n'], 0.5)
           #log_n = ml_deltas['n']
           log_ls = pm.MvNormal('log_ls', mu=np.log(ml_deltas['ls']), cov = np.eye(n_dim)*2, shape=(n_dim,))
           
           s = pm.Deterministic('s', tt.exp(log_s))
           ls = pm.Deterministic('ls', tt.exp(log_ls))
           n = pm.Deterministic('n', tt.exp(log_n))
           
           bias = pm.Normal('b', 0, 1)
           
           # Specify the covariance function
       
           cov_main = pm.gp.cov.Constant(s**2)*pm.gp.cov.ExpQuad(n_dim, ls) + pm.gp.cov.Constant(bias)
           cov_noise = pm.gp.cov.WhiteNoise(n**2)
       
           gp_main = pm.gp.Marginal(cov_func=cov_main)
           gp_noise = pm.gp.Marginal(cov_func=cov_noise) 
           
           gp = gp_main
           
           trace_prior = pm.sample(500)
           
      with physics_model:
            
           y_ = gp.marginal_likelihood("y", X=X_train, y=y_train, noise=cov_noise)
           trace_hmc = pm.sample(draws=500, tune=300, chains=2)
           
      with physics_model:
    
            pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc_2/', overwrite=True)
      
      with physics_model:
      
            trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/')
        
      with physics_model:
            
            mf = pm.ADVI()
      
            tracker_mf = pm.callbacks.Tracker(
            mean = mf.approx.mean.eval,    
            std = mf.approx.std.eval)
           
            mf.fit(n=50000, callbacks=[tracker_mf])
            
            trace_mf = mf.approx.sample(2000)
      
      with physics_model:
            
            fr = pm.FullRankADVI(start=ml_deltas)
              
            tracker_fr = pm.callbacks.Tracker(
            mean = fr.approx.mean.eval,    
            std = fr.approx.std.eval)
            
            fr.fit(n=50000, callbacks=[tracker_fr])
            trace_fr = fr.approx.sample(2000)
            
            
            bij_mf = mf.approx.groups[0].bij 
            mf_param = {param.name: bij_mf.rmap(param.eval()) for param in mf.approx.params}
      
            bij_fr = fr.approx.groups[0].bij
            fr_param = {param.name: bij_fr.rmap(param.eval()) for param in fr.approx.params}
            
      trace_prior_df = pm.trace_to_dataframe(trace_prior)
      trace_hmc_df = pm.trace_to_dataframe(trace_hmc)
      
      # Marginal posteriors
      
      multid_traceplot(trace_hmc_df, varnames_log_unravel, feature_mapping, ml_deltas_log)
      
      # Prior Posterior Learning 
      
      pa.plot_prior_posterior_plots(trace_prior_df, trace_hmc_df, varnames_log_unravel, ml_deltas_log, 'Prior Posterior HMC')

       # Forest plot
      
      pm.forestplot(trace_hmc, varnames=varnames, rhat=True, quartiles=False)
      plt.title('95% Credible Intervals (HMC)', fontsize='small')

      pm.forestplot(trace_mf, varnames=varnames, rhat=True, quartiles=False)
      plt.title('95% Credible Intervals (Mean-Field VI)', fontsize='small')
      
      pm.forestplot(trace_fr, varnames=varnames, rhat=True, quartiles=False)
      plt.title('95% Credible Intervals (Full-Rank VI)', fontsize='small')

      # Convergence 
   
      ad.convergence_report(tracker_mf, mf.hist, varnames_log_unravel, 'Mean-Field Convergence')
      ad.convergence_report(tracker_fr, fr.hist, varnames_log_unravel, 'Full-Rank Convergence')
      
      # Pair scatter plot 
      from itertools import combinations

      bi_list = []
      for i in combinations(varnames_unravel, 2):
            bi_list.append(i)
            
      for i, j  in zip(bi_list, np.arange(len(bi_list))):
        print(i)
        print(j)
        if np.mod(j,8) == 0:
            fig = plt.figure(figsize=(15,8))
        plt.subplot(2,4,np.mod(j, 8)+1)
        sns.scatterplot(trace_hmc_df[i[0]], trace_hmc_df[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        plt.scatter(ml_deltas_unravel[i[0]], ml_deltas_unravel[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
        
      # Predictions

      # HMC
      
      pa.write_posterior_predictive_samples(trace_hmc, 10, X_test, results_path + 'pred_dist/', method='hmc_sub', gp=gp) 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc_sub.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc_sub.csv')
      
     # sample_means_hmc = forward_mu(sample_means_hmc, emp_mu, emp_std)
      #sample_stds_hmc = forward_std(sample_stds_hmc, emp_std)
      
      mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
      
      # persist mu_hmc
      
      np.savetxt(fname=results_path + 'pred_dist/' + 'means_hmc.csv', X=mu_hmc, delimiter=',', header='')
      
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
      
      rmse_hmc = pa.rmse(mu_hmc, y_test)
      se_rmse_hmc = pa.se_of_rmse(mu_hmc, y_test)
      lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
      
      print('rmse_hmc:' + str(rmse_hmc))
      print('se_rmse_hmc:' + str(se_rmse_hmc))
      print('lpd_hmc:' + str(lpd_hmc))
      
      # MF
      
      pa.write_posterior_predictive_samples(trace_mf, 50, X_test, results_path + 'pred_dist/', method='mf', gp=gp) 
      
      sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
      sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')
      
      #sample_means_mf = forward_mu(sample_means_mf, emp_mu, emp_std)
      #sample_stds_mf = forward_std(sample_stds_mf, emp_std)
      
      mu_mf = pa.get_posterior_predictive_mean(sample_means_mf)
      
       # persist mu_mf
      
      np.savetxt(fname=results_path + 'pred_dist/' + 'means_mf.csv', X=mu_mf, delimiter=',', header='')
      
      lower_mf, upper_mf = pa.get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)
      
      rmse_mf = pa.rmse(mu_mf, y_test)
      se_rmse_mf = pa.se_of_rmse(mu_mf, y_test)
      lppd_mf, lpd_mf = pa.log_predictive_mixture_density(y_test, sample_means_mf, sample_stds_mf)

      print('rmse_mf:' + str(rmse_mf))
      print('se_rmse_mf:' + str(se_rmse_mf))
      print('lpd_mf:' + str(lpd_mf))

      # FR
      
      pa.write_posterior_predictive_samples(trace_fr, 50, X_test, results_path +  'pred_dist/', method='fr', gp=gp) 
      
      sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr.csv')
      sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')
      
      #sample_means_fr = forward_mu(sample_means_fr, emp_mu, emp_std)
      #sample_stds_fr = forward_std(sample_stds_fr, emp_std)
      
      mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
      
      # persist mu_fr
      
      np.savetxt(fname=results_path + 'pred_dist/' + 'means_fr.csv', X=mu_fr, delimiter=',', header='')
      
      lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
      rmse_fr = pa.rmse(mu_fr, y_test)
      se_rmse_fr = pa.se_of_rmse(mu_fr, y_test)
      lppd_fr, lpd_fr = pa.log_predictive_mixture_density(y_test, sample_means_fr, sample_stds_fr)
      
      print('rmse_fr:' + str(rmse_fr))
      print('se_rmse_fr:' + str(se_rmse_fr))
      print('lpd_fr:' + str(lpd_fr))
      

from itertools import combinations

trace = trace_hmc_df
varnames = varnames_unravel
ml_deltas = ml_deltas_unravel

bi_list = []
for i in combinations(varnames, 2):
      bi_list.append(i)
            
bi_list_sub = bi_list[0:20]

for i, j  in zip(bi_list_sub, np.arange(len(bi_list_sub))):
      print(i)
      print(j)
      if np.mod(j,8) == 0:
            plt.figure(figsize=(15,8))
      plt.subplot(2,4,np.mod(j, 8)+1)
      #g = sns.kdeplot(trace[i[0]], trace[i[1]], color="b", shade=True, alpha=0.7, bw='silverman')
      plt.scatter(trace[i[0]], trace[i[1]], s=0.5, color='b', alpha=0.4)
      plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
      plt.xlabel(i[0])
      plt.ylabel(i[1])
      plt.tight_layout()     
      
      