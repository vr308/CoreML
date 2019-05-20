#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:36:52 2019

@author: vidhi
"""

import numpy as np 
import matplotlib.pylab as plt
import theano.tensor as tt
import pymc3 as pm
import pandas as pd
from sampled import sampled
from scipy.misc import derivative
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from sympy import symbols, diff, exp, log, power


def se_kernel(sigma_f, ls, sigma_n, x1, x2):
      
      return sigma_f**2*exp(-0.5*(1/ls**2)*(x1 - x2)**2) + sigma_n**2

sigma_f, ls, sigma_n, x1, x2 = symbols('sigma_f ls sigma_n x1 x2', real=True)

dk_dsf = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_f)
dk_dls = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), ls)
dk_dsn = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_n)

d2k_d2sf = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_f, 2)
d2k_d2ls = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), ls, 2)
d2k_d2sn = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_n, 2)

d2k_dsfdls = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_f, ls)
d2k_dsfdsn = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), sigma_f, sigma_n)
d2k_dlsdsn = diff(se_kernel(sigma_f, ls, sigma_n, x1, x2), ls, sigma_n)

def gradient_gp_pred_mean(x_star, K_inv, y, mu_theta, cov_theta):
      
      return 


def curvature_gp_pred_mean(x_star, K_inv, y, mu_theta, cov_thet):
      
      return


def gradient_gp_pred_var(x_star, K_inv, mu_theta, cov_thet):
      
      return 

def curvature_gp_pred_var(x_star, K_inv, mu_theta, cov_thet):
      
      return 


def load_datasets(path, n_train):
      
       n_test = 200 - n_train
       X = np.asarray(pd.read_csv(path + 'X_' + str(n_train) + '.csv', header=None)).reshape(n_train,1)
       y = np.asarray(pd.read_csv(path + 'y_' + str(n_train) + '.csv', header=None)).reshape(n_train,)
       X_star = np.asarray(pd.read_csv(path + 'X_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,1)
       f_star = np.asarray(pd.read_csv(path + 'f_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,)
       return X, y, X_star, f_star
 
      
def get_ml_report(X, y, X_star, f_star):
      
          kernel = Ck(50, (1e-10, 1e3)) * RBF(0.0001, length_scale_bounds=(1e-10, 8)) + WhiteKernel(1.0, noise_level_bounds=(1e-10,1000))
          
          gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
              
          # Fit to data 
          gpr.fit(X, y)        
          ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
          post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
          post_std = np.sqrt(np.diag(post_cov))
          post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
          post_samples = np.random.multivariate_normal(post_mean, post_cov , 10)
          rmse_ = rmse(post_mean, f_star)
          lpd_ = -log_predictive_density(f_star, post_mean, post_std)
          title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + '-LPD: ' + str(lpd_)     
          ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
                            'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
          return gpr, post_mean, post_std, post_std_nf, rmse_, lpd_, ml_deltas_dict, title
 
# Generative model for full Bayesian treatment

@sampled
def generative_model(X, y):
      
       # prior on lengthscale 
       log_ls = pm.Uniform('log_ls', lower=-5, upper=5)
       ls = pm.Deterministic('ls', tt.exp(log_ls))
       
        #prior on noise variance
       log_n = pm.Uniform('log_n', lower=-5, upper=5)
       noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))
         
       #prior on signal variance
       log_s = pm.Uniform('log_s', lower=-5, upper=5)
       sig_sd = pm.Deterministic('sig_sd', tt.exp(log_s))
       
       # Specify the covariance function.
       cov_func = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)
       

raw_mapping = {'log_ls_interval__':  generative_model(X=X_40, y=y_40).log_ls_interval__, 
              'log_s_interval__': generative_model(X=X_40, y=y_40).log_s_interval__, 
              'log_n_interval__':  generative_model(X=X_40, y=y_40).log_n_interval__}

name_mapping = {'log_ls_interval__':  'ls', 
              'log_s_interval__': 'sig_sd', 
              'log_n_interval__':  'noise_sd'}

rev_name_mapping = {'ls' : 'log_ls_interval__', 
              'sig_sd' : 'log_s_interval__', 
               'noise_sd': 'log_n_interval__'}


def update_param_dict(model, param_dict, summary_trace):
      
      keys = list(param_dict['mu'].keys())
      
      # First tackling transformed means
      
      mu_implicit = {}
      rho_implicit = {}
      for i in keys:
            if (i[-2:] == '__'):
                  name = name_mapping[i]
                  mean_value = np.exp(raw_mapping.get(i).distribution.transform_used.backward(param_dict['mu'][i]).eval())
                  sd_value = summary_trace['sd'][name]
                  mu_implicit.update({name : mean_value})
                  rho_implicit.update({name : sd_value})
      param_dict.update({'mu_implicit' : mu_implicit})
      param_dict.update({'rho_implicit' : rho_implicit})
      return param_dict
             
def transform_tracker_values(tracker, param_dict):

      mean_df = pd.DataFrame(np.array(tracker['mean']), columns=list(param_dict['mu'].keys()))
      sd_df = pd.DataFrame(np.array(tracker['std']), columns=list(param_dict['mu'].keys()))
      for i in mean_df.columns:
            print(i)
            if (i[-2:] == '__'):
                 mean_df[name_mapping[i]] = np.exp(raw_mapping.get(i).distribution.transform_used.backward(mean_df[i]).eval()) 
      return mean_df, sd_df

def convergence_report(tracker_mf, tracker_fr, true_hyp, varnames):
      
      # Plot Negative ElBO track with params in true space
      
      mean_mf_df, sd_mf_df = transform_tracker_values(tracker_mf, mf_param)
      mean_fr_df, sd_fr_df = transform_tracker_values(tracker_fr, fr_param)

      fig = plt.figure(figsize=(16, 9))
      for i in np.arange(3):
            print(i)
            plt.subplot(2,3,i+1)
            plt.plot(mean_mf_df[varnames[i]], color='coral')
            plt.plot(mean_fr_df[varnames[i]], color='green')
            plt.title(varnames[i], fontsize='x-small')
            plt.axhline(true_hyp[varnames[i]], color='r', alpha=0.4)
      for i in [4,5,6]:
            print(i)
            plt.subplot(2,3,i)
            plt.plot(sd_mf_df[rev_name_mapping[varnames[i-4]]], color='coral')
            plt.plot(sd_fr_df[rev_name_mapping[varnames[i-4]]], color='green')
            plt.title(varnames[i-4], fontsize='small')
      plt.suptitle('Convergence Mean-Field vs. Full-Rank', fontsize='x-small')


def plot_elbo_convergence(mf, fr):
      
      fig = plt.figure(figsize=(16, 9))
      mf_ax = fig.add_subplot(211)
      fr_ax = fig.add_subplot(212)
      mf_ax.plot(mf.hist)
      fr_ax.plot(fr.hist)
      mf_ax.set_title('Mean-Field')
      fr_ax.set_title('Full Rank')
      fig.suptitle('Negative ELBO Track')
      
      
def get_implicit_variational_posterior(var, means, std, x):
      
      sigmoid = lambda x : 1 / (1 + np.exp(-x))
      eps = lambda x : var.distribution.transform_used.forward_val(np.log(x))
      backward_theta = lambda x: var.distribution.transform_used.backward(x).eval()   
      width = (var.distribution.transform_used.b -  var.distribution.transform_used.a).eval()
      total_jacobian = lambda x: x*(width)*sigmoid(eps(x))*(1-sigmoid(eps(x)))
      pdf = lambda x: st.norm.pdf(eps(x), means[var.name], std[var.name])/total_jacobian(x)
      return pdf(x)
      
      
def trace_report(mf, fr, means_mf, std_mf, means_fr, std_fr, true_hyp, trace_mf, trace_fr):
      
      l_int = np.linspace(min(trace_fr['ls'])-1, max(trace_fr['ls'])+1, 1000)
      n_int = np.linspace(min(trace_fr['noise_sd'])-1, max(trace_fr['noise_sd'])+1, 1000)
      s_int = np.linspace(min(trace_fr['sig_sd'])-1, max(trace_fr['sig_sd'])+1, 1000)
      
      ranges = [s_int, l_int, n_int]
      
      mf_rv = [mf.approx.model.log_s_interval__, mf.approx.model.log_ls_interval__, mf.approx.model.log_n_interval__]
      fr_rv = [fr.approx.model.log_s_interval__, fr.approx.model.log_ls_interval__, fr.approx.model.log_n_interval__]
      
      fig = plt.figure(figsize=(14, 5))
      for i in np.arange(3):
            print(i)
            plt.subplot(1,3,i+1)
            plt.plot(ranges[i], get_implicit_variational_posterior(mf_rv[i], means_mf, std_mf, ranges[i]), color='coral', label='MF') 
            plt.hist(trace_fr[varnames[i]], bins=100, normed=True, color='g', alpha=0.3)
            plt.hist(trace_mf[varnames[i]], bins=100, normed=True, color='coral', alpha=0.4)
            plt.plot(ranges[i], get_implicit_variational_posterior(fr_rv[i], means_fr, std_fr, ranges[i]), color='g', label='FR')
            plt.title(varnames[i], fontsize='small')
            plt.legend(fontsize='x-small')
      plt.suptitle('Marginal Posterior', fontsize='x-small')
      
# Pair grid catalog

bi_list = []
for i in combinations(varnames, 2):
      bi_list.append(i)

def get_bivariate_hyp(bi_list, trace_mf, trace_fr):
      
      fig = plt.figure(figsize=(14,5))
      for i, j  in zip(bi_list, np.arange(len(bi_list))):
              plt.subplot(1,3,j+1)
              sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=0.9)
              sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
              #plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
              plt.xlabel(i[0])
              plt.ylabel(i[1])
              plt.tight_layout()

if __name__ == "__main__":

   
      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/1d/'
      home_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/1d/'

      # Edit here to change generative model
      
      input_dist =  'Unif'
      snr = 2
      suffix = input_dist + '/' + 'SNR_' + str(snr) + '/'
      true_hyp = {'sig_sd' : np.round(np.sqrt(100),3), 'ls' : 5, 'noise_sd' : np.round(np.sqrt(50),3)}

      data_path = uni_path + suffix
      
      varnames = ['sig_sd', 'ls', 'noise_sd']
      log_varnames = ['log_s', 'log_ls', 'log_n']
    
      X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40)
      
      # ML 
      
      gpr_40, pp_mean_ml_40, pp_std_ml_40, pp_std_ml_nf_40, rmse_ml_40, lpd_ml_40, ml_deltas_dict_40, title_40 =  get_ml_report(X_40, y_40, X_star_40, f_star_40)
     
      
      # VI - Mean Field and Full-Rank
      
      with generative_model(X=X_40, y=y_40):
             
            mf = pm.ADVI()
            fr = pm.FullRankADVI()
      
            tracker_mf = pm.callbacks.Tracker(
            mean = mf.approx.mean.eval,    
            std = mf.approx.std.eval)
            
            tracker_fr = pm.callbacks.Tracker(
            mean = fr.approx.mean.eval,    
            std = fr.approx.std.eval)
      
            mf.fit(callbacks=[tracker_mf], n=25000)
            fr.fit(callbacks=[tracker_fr], n=25000)
            
            trace_mf = mf.approx.sample(5000)
            trace_fr = fr.approx.sample(5000)
          
      
      # Post-processing 

      trace_mf_df = pm.trace_to_dataframe(trace_mf)
      trace_fr_df = pm.trace_to_dataframe(trace_fr)
      
      means_mf = mf.approx.bij.rmap(mf.approx.mean.eval())  
      std_mf = mf.approx.bij.rmap(mf.approx.std.eval())  
      
      means_fr = fr.approx.bij.rmap(fr.approx.mean.eval())  
      std_fr = fr.approx.bij.rmap(fr.approx.std.eval())  
      
      
      bij_mf = mf.approx.groups[0].bij
      mf_param = {param.name: bij_mf.rmap(param.eval())
      	 for param in mf.approx.params}
      
      bij_fr = fr.approx.groups[0].bij
      fr_param = {param.name: bij_fr.rmap(param.eval())
      	 for param in fr.approx.params}
      
      
      # Variational parameters track  
      
      convergence_report(tracker_mf, tracker_fr, true_hyp, varnames)

      # ELBO Convergence
      
      plot_elbo_convergence(mf, fr)
      
      # Traceplot with Mean-Field and Full-Rank
      
      trace_report(mf, fr, means_mf, std_mf, means_fr, std_fr, true_hyp, trace_mf, trace_fr)
      
      # Bi-variate relationship
      
      get_bivariate_hyp(bi_list, trace_mf, trace_fr)
      
      # Predictions with MCMC
      
      
      
      # Full deterministic prediction
      
      mu_theta = {'sigma_f': , 'ls': , 'sigma_n': }
      
      
