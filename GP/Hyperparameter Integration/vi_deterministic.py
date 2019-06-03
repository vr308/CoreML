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
from theano.tensor.nlinalg import matrix_inverse
import pandas as pd
from sampled import sampled
from scipy.misc import derivative
import csv
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from sympy import symbols, diff, exp, log, power


def se_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*exp(-0.5*(1/ls**2)*(x1 - x2)**2) + noise_sd**2

def gradient_K(X):
      
      n_train = len(X) 
      
      dK_dsf_m = np.zeros(shape=(n_train, n_train))
      dK_dls_m = np.zeros(shape=(n_train, n_train))
      dK_dsn_m = np.zeros(shape=(n_train, n_train))
      
      i, j = np.meshgrid(np.arange(n_train), np.arange(n_train))
      index = np.vstack([j.ravel(), i.ravel()]).T
      
      for h in index:
            dK_dsf_m[h[0]][h[1]] = dk_dsf.subs({x1: X[h[0]], x2: X[h[1]]})
            dK_dls_m[h[0]][h[1]] = dk_dls.subs({x1: X[h[0]], x2: X[h[1]]})
            dK_dsn_m[h[0]][h[1]] = dk_dsn.subs({x1: X[h[0]], x2: X[h[1]]})
      
      return np.array([dK_dsf_m, dK_dls_m, dK_dsn_m]) 

def curvature_K(X):
      
      # Rank 4 tensor  -> 3 x 3 x n x n 
      
      d2K_d2sf_m = np.zeros(shape=(n_train, n_train))
      d2K_d2ls_m = np.zeros(shape=(n_train, n_train))
      d2K_d2sn_m = np.zeros(shape=(n_train, n_train))
      d2K_dls_dsf_m = np.zeros(shape=(n_train, n_train))
      d2K_dls_dsn_m = np.zeros(shape=(n_train, n_train))
      d2K_dsn_dsf_m = np.zeros(shape=(n_train, n_train))
      
      i, j = np.meshgrid(np.arange(n_train), np.arange(n_train))
      index = np.vstack([j.ravel(), i.ravel()]).T
      
      for h in index:
          
            d2K_d2sf_m[h[0]][h[1]] = d2k_d2sf.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_d2ls_m[h[0]][h[1]] = d2k_d2ls.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_d2sn_m[h[0]][h[1]] = d2k_d2sn.subs({x1: X[h[0]], x2: X[h[1]]})
            
            d2K_dls_dsf_m[h[0]][h[1]] = d2k_dsfdls.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_dls_dsn_m[h[0]][h[1]] = d2k_dlsdsn.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_dsn_dsf_m[h[0]][h[1]] = d2k_dsfdsn.subs({x1: X[h[0]], x2: X[h[1]]})

      T1 = [d2K_d2sf_m, d2K_dls_dsf_m, d2K_dsn_dsf_m]  
      T2 = [d2K_dls_dsf_m, d2K_d2ls_m, d2K_dls_dsn_m]
      T3 = [d2K_dsn_dsf_m, d2K_dls_dsn_m, d2K_d2sn_m]
      
      return np.array([T1, T2, T3])

def gradient_K_star(X, x_star):
      
      n = len(X)
      row1 = np.zeros(shape=(n))
      row2 = np.zeros(shape=(n))
      row3 = np.zeros(shape=(n))
      
      for i in np.arange(n):
            row1[i] = dk_dsf.subs({x1: X[i], x2: x_star})
            row2[i] = dk_dls.subs({x1: X[i], x2: x_star})
            #row3[i] = dk_dsn.subs({x1: X[i], x2: x_star})
            row3[i] = 0
            
      return np.array([row1, row2, row3]).T

def curvature_K_star(X, x_star):
      
      # Rank 3 tensor  -> 3 x 3 x n 
      
     row_d2sf = np.zeros(shape=(n_train))
     row_d2ls = np.zeros(shape=(n_train))
     row_d2sn = np.zeros(shape=(n_train))
     
     row_dsfdls = np.zeros(shape=(n_train))
     row_dlsdsn = np.zeros(shape=(n_train))
     row_dsfdsn = np.zeros(shape=(n_train))
     
     for h in np.arange(len(X)):
           
           row_d2sf[h] = d2k_d2sf.subs({x1: X[h], x2: x_star})
           row_d2ls[h] = d2k_d2ls.subs({x1: X[h], x2: x_star})
           #row_d2sn[h] = d2k_d2sn.subs({x1: X[h], x2: x_star})
           row_d2sn[h] = 0
           
           row_dsfdls[h] = d2k_dsfdls.subs({x1: X[h], x2: x_star})
           row_dlsdsn[h] = d2k_dlsdsn.subs({x1: X[h], x2: x_star})
           row_dsfdsn[h] = d2k_dsfdsn.subs({x1: X[h], x2: x_star})
           
     M1 = np.array([row_d2sf, row_dsfdls, row_dsfdsn])
     M2 = np.array([row_dsfdls, row_d2ls, row_dlsdsn])
     M3 = np.array([row_dsfdsn, row_dlsdsn, row_d2sn])
      
     return np.array([M1, M2, M3]).T

def gradient_K_star_star(x_star):
      
      #return np.array([dk_dsf.subs({x1: x_star, x2: x_star}), dk_dls.subs({x1: x_star, x2: x_star}), dk_dsn.subs({x1: x_star, x2: x_star})])
      return np.array([dk_dsf.subs({x1: x_star, x2: x_star}), dk_dls.subs({x1: x_star, x2: x_star}), 0.0], dtype=np.float)

def curvature_K_star_star(x_star):
      
      # 3 X 3 matrix
      row1 = [d2k_d2sf.subs({x1: x_star, x2: x_star}), d2k_dsfdls.subs({x1: x_star, x2: x_star}), d2k_dsfdsn.subs({x1: x_star, x2: x_star})]
      row2 = [d2k_dsfdls.subs({x1: x_star, x2: x_star}), d2k_d2ls.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star})]
      #row3 = [d2k_dsfdsn.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star}), d2k_d2sn.subs({x1: x_star, x2: x_star})]
      row3 = [0, 0, 0]
      
      return np.array([row1, row2, row3], dtype=np.float)

def gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s):
      
      dKsT = gradient_K_star(X, x_star).T
      
      return np.matmul(np.matmul(dKsT, K_inv), y)[:, np.newaxis] + np.matmul(np.matmul(K_s.T, dK_inv), y)

def curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv):
      
      dKsT = gradient_K_star(X, x_star).T
      d2KsT = curvature_K_star(X, x_star).T
      
      return  np.matmul(np.matmul(d2KsT, K_inv),y) + 2*np.matmul(np.matmul(dKsT, dK_inv), y) + np.matmul(np.matmul(K_s.T, d2K_inv),y).reshape(3,3).T

def curvature_gp_pred_var(X, x_star, y, K_s,  K_inv, dK_inv, d2K_inv):
      
      dK_starT = gradient_K_star(X, x_star).T
      d2K_star_star = curvature_K_star_star(x_star)
      d2K_starT = curvature_K_star(X, x_star).T
      
      J = curvature_K_star(X, x_star)
      
      d2K_starTT = np.array([J[:,:,0], J[:,:,1], J[:,:,2]])
      
      return  d2K_star_star - np.matmul(np.matmul(d2K_starT, K_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(dK_starT,dK_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(dK_starT, K_inv), dK_starT.T) -  np.matmul(np.matmul(K_s.T, d2K_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(K_s.T, dK_inv), dK_starT.T).reshape(3,3) - np.matmul(np.matmul(K_s.T,K_inv), d2K_starTT).reshape(3,3)

def deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta):
      
      K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      pred_vi_mean = np.matmul(np.matmul(K_s.T.eval(), K_inv), y)
      
      d2_gp_mean = curvature_gp_pred_mean(X, x_star, y, K_s.eval(), K_inv, dK_inv, d2K_inv) # 3x3 matrix
      
      return pred_vi_mean + 0.5*np.trace(np.matmul(d2_gp_mean, cov_theta))

def deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta):
      
      K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      #pred_vi_var =  K_ss.eval() - np.matmul(np.matmul(K_s.T.eval(), K_inv), K_s.eval())
      pred_vi_mean, pred_vi_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
      
      d1_gp_mean = gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s.eval())
      d2_gp_var = curvature_gp_pred_var(X, x_star, y, K_s.eval(), K_inv, dK_inv, d2K_inv)
      
      return pred_vi_std**2 + 0.5*np.trace(np.matmul(d2_gp_var, cov_theta)) + np.trace(np.matmul(np.matmul(d1_gp_mean, d1_gp_mean.T), cov_theta))


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
    
def get_star_kernel_matrix_blocks(X, X_star, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          return K_s, K_ss

def get_kernel_matrix_blocks(X, X_star, n_train, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K = cov(X)
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          K_noise = K + np.square(point['noise_sd'])*tt.eye(n_train)
          K_inv = matrix_inverse(K_noise)
          return K, K_s, K_ss, K_noise, K_inv

def analytical_gp(y, K, K_s, K_ss, K_noise, K_inv):
    
          L = np.linalg.cholesky(K_noise.eval())
          alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
          v = np.linalg.solve(L, K_s.eval())
          post_mean = np.dot(K_s.eval().T, alpha)
          post_cov = K_ss.eval() - v.T.dot(v)
          post_std = np.sqrt(np.diag(post_cov))
          return post_mean,  post_std
    
def compute_log_marginal_likelihood(K_noise, y):
      
      return np.log(st.multivariate_normal.pdf(y, cov=K_noise.eval()))

def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds, weights):
      
      # Fixed at 95% CI
      
      prob = weights/np.sum(weights)
      
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True, p=prob)
            mixture_draws = np.array([st.norm.rvs(loc=sample_means.iloc[j,i], scale=sample_stds.iloc[j,i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5,97.5])
            lower_.append(lower)
            upper_.append(upper)
      return np.array(lower_), np.array(upper_)

def get_posterior_predictive_mean(sample_means, weights):
      
      if weights is None:
            return np.mean(sample_means)
      else:
            return np.average(sample_means, axis=0, weights=weights)
                         
#--------------------Predictive performance metrics-------------------------
      
def rmse(post_mean, f_star):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - f_star))),3)

def log_predictive_density(f_star, post_mean, post_std):
      
      lppd_per_point = []
      for i in np.arange(len(f_star)):
            lppd_per_point.append(st.norm.pdf(f_star[i], post_mean[i], post_std[i]))
      #lppd_per_point.remove(0.0)
      return np.round(np.mean(np.log(lppd_per_point)),3)

def log_predictive_mixture_density(f_star, list_means, list_std, weights):
            
      lppd_per_point = []
      for i in np.arange(len(f_star)):
            print(i)
            components = []
            for j in np.arange(len(list_means)):
                  components.append(st.norm.pdf(f_star[i], list_means.iloc[:,i][j], list_std.iloc[:,i][j]))
            lppd_per_point.append(np.average(components, weights=weights))
      return lppd_per_point, np.round(np.mean(np.log(lppd_per_point)),3)
    

def write_posterior_predictive_samples(trace, thin_factor, X, y, X_star, path, method):
      
      means_file = path + 'means_' + method + '_' + str(len(X)) + '.csv'
      std_file = path + 'std_' + method + '_' + str(len(X)) + '.csv'
      trace_file = path + 'trace_' + method + '_' + str(len(X)) + '.csv'
    
      means_writer = csv.writer(open(means_file, 'w')) 
      std_writer = csv.writer(open(std_file, 'w'))
      trace_writer = csv.writer(open(trace_file, 'w'))
      
      means_writer.writerow(X_star.flatten())
      std_writer.writerow(X_star.flatten())
      trace_writer.writerow(varnames + ['lml'])
      
      for i in np.arange(len(trace))[::thin_factor]:
            
            print('Predicting ' + str(i))
            K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), trace[i])
            post_mean, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
            marginal_likelihood = compute_log_marginal_likelihood(K_noise, y)
            #mu, var = pm.gp.Marginal.predict(Xnew=X_star, point=trace[i], pred_noise=False, diag=True)
            #std = np.sqrt(var)
            list_point = [trace[i]['sig_sd'], trace[i]['ls'], trace[i]['noise_sd'], marginal_likelihood]
            
            print('Writing out ' + str(i) + ' predictions')
            means_writer.writerow(np.round(post_mean, 3))
            std_writer.writerow(np.round(post_std, 3))
            trace_writer.writerow(np.round(list_point, 3))

def load_post_samples(means_file, std_file):
      
      return pd.read_csv(means_file, sep=',', header=0), pd.read_csv(std_file, sep=',', header=0)
 
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
       

raw_mapping = {'log_ls_interval__':  generative_model(X=X_20, y=y_20).log_ls_interval__, 
              'log_s_interval__': generative_model(X=X_20, y=y_20).log_s_interval__, 
              'log_n_interval__':  generative_model(X=X_20, y=y_20).log_n_interval__}

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
                  sd_value = summary_trace['sd'][name]
                  #mean_value = np.exp(raw_mapping.get(i).distribution.transform_used.backward(param_dict['mu'][i]).eval())
                  mean_value = summary_trace['mean'][name]
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

def plot_vi_mf_fr_ml_joint(pp_mean_ml, pp_std_ml_nf, pp_mean_mf, pp_mean_fr, lower_mf, upper_mf, lower_fr, upper_fr):
             
             plt.figure()
             plt.plot(X_star, pp_mean_ml, color='r', label='ML')
             plt.plot(X_star, pp_mean_mf, color='coral',label='MF')
             plt.plot(X_star, pp_mean_fr, color='g', label='FR')
             plt.fill_between(X_star.ravel(), pp_mean_ml - 1.96*pp_std_ml_nf, pp_mean_ml + 1.96*pp_std_ml_nf, color='r', alpha=0.3)
             plt.fill_between(X_star.ravel(), lower_mf, upper_mf, color='coral', alpha=0.3)
             plt.fill_between(X_star.ravel(), lower_fr, upper_fr, color='g', alpha=0.3)
             plt.legend(fontsize='small')
             plt.title('ML vs. MF vs. FR', fontsize='small')
             

if __name__ == "__main__":

   
      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      mac_path = '/Users/vidhi.lalchand/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      desk_home_path = '/home/vr308/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      
      path = uni_path


      # Edit here to change generative model
      
      input_dist =  'Unif'
      snr = 2
      suffix = input_dist + '/' + 'SNR_' + str(snr) + '/'
      true_hyp = {'sig_sd' : np.round(np.sqrt(100),3), 'ls' : 5, 'noise_sd' : np.round(np.sqrt(50),3)}

      data_path = path + 'Data/1d/' + suffix
      results_path = path + 'Results/1d/' + suffix 
      
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
      
      update_param_dict(mf, mf_param, pm.summary(trace_mf))
      update_param_dict(fr, fr_param, pm.summary(trace_fr))
      
      # Variational parameters track  
      
      convergence_report(tracker_mf, tracker_fr, true_hyp, varnames)

      # ELBO Convergence
      
      plot_elbo_convergence(mf, fr)
      
      # Traceplot with Mean-Field and Full-Rank
      
      trace_report(mf, fr, means_mf, std_mf, means_fr, std_fr, true_hyp, trace_mf, trace_fr)
      
      # Bi-variate relationship
      
      bi_list = []
      for i in combinations(varnames, 2):
          bi_list.append(i)

      
      get_bivariate_hyp(bi_list, trace_mf, trace_fr)
      
      # Predictions with MCMC
      
      thin_factor=100
      write_posterior_predictive_samples(trace_mf, thin_factor, X_40, y_40, X_star_40, results_path, 'mf')
      write_posterior_predictive_samples(trace_fr, thin_factor, X_40, y_40, X_star_40, results_path, 'fr')

      post_means_mf_40, post_stds_mf_40 = load_post_samples(results_path + 'means_mf_40.csv', results_path + 'std_mf_40.csv')
      post_means_fr_40, post_stds_fr_40 = load_post_samples(results_path + 'means_fr_40.csv', results_path + 'std_fr_40.csv')
       
      u40 = np.ones(len(post_means_mf_40))

      pp_mean_mf_40 = get_posterior_predictive_mean(post_means_mf_40, weights=None)
      pp_mean_fr_40 = get_posterior_predictive_mean(post_means_fr_40, weights=None)
       
      lower_mf_40, upper_mf_40 = get_posterior_predictive_uncertainty_intervals(post_means_mf_40, post_stds_mf_40, u40)
      lower_fr_40, upper_fr_40 = get_posterior_predictive_uncertainty_intervals(post_means_fr_40, post_stds_fr_40, u40)
       
      rmse_mf_40 = rmse(pp_mean_mf_40, f_star_40)
      rmse_fr_40 = rmse(pp_mean_fr_40, f_star_40)

      lppd_mf_40, lpd_mf_40 = log_predictive_mixture_density(f_star_40, post_means_mf_40, post_stds_mf_40, None)
      lppd_fr_40, lpd_fr_40 = log_predictive_mixture_density(f_star_40, post_means_fr_40, post_stds_fr_40, None)

      title_mf_40 = 'RMSE: ' + str(rmse_mf_40) + '\n' + '-LPD: ' + str(-lpd_mf_40)
      title_fr_40 = 'RMSE: ' + str(rmse_fr_40) + '\n' + '-LPD: ' + str(-lpd_fr_40)
       
      plot_vi_mf_fr_ml_joint(pp_mean_ml_40, pp_std_ml_nf_40, pp_mean_mf_40, pp_mean_fr_40, lower_mf_40, upper_mf_40, lower_fr_40, upper_fr_40)
      
      # Full deterministic prediction
      
       # Extracting mu_theta and cov_theta
      
      cov_theta_mf =  np.cov(trace_mf_df[varnames], rowvar=False)
      cov_theta_fr = np.cov(trace_fr_df[varnames], rowvar=False)
      
      mu_theta_mf = mf_param['mu_implicit']
      mu_theta_fr = fr_param['mu_implicit']
      
      sig_sd, ls, noise_sd, x1, x2 = symbols('sig_sd ls noise_sd x1 x2', real=True)
      
      mu_theta = mu_theta_mf

      dk_dsf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_d2sf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2ls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2sn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_dsfdls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dsfdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dlsdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      n_train = 40 
            
      cov = pm.gp.cov.Constant(mu_theta['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=mu_theta['ls'])
      
      K = cov(X).eval()
      K_noise = (K + np.square(mu_theta['noise_sd'])*tt.eye(n_train))
      K_inv = matrix_inverse(K_noise).eval()
         
      dK = gradient_K(X)
      d2K = curvature_K(X)
      
      dK_inv = -np.matmul(np.matmul(K_inv, dK),K_inv)
      d2K_inv = -np.matmul(np.matmul(dK_inv, dK), K_inv) - np.matmul(np.matmul(K_inv, d2K), K_inv) - np.matmul(np.matmul(K_inv, dK), dK_inv)
            
      vi_pred_mean_mf = []
      vi_pred_var_mf = []

      vi_pred_mean_fr = []
      vi_pred_var_fr = []
      
      vi_pred_mean_mf_mode = np.matmul(np.matmul(cov(X, X_star).T.eval(), K_inv), y)
      vi_pred_var_mf_mode =  np.diag(cov(X_star, X_star).eval() - np.matmul(np.matmul(cov(X, X_star).T.eval(), K_inv), cov(X, X_star).eval()))
                  
      for i in np.arange(len(X_star)):
          
          print(i)
          x_star = X_star[i].reshape(1,1)
          vi_pred_mean_mf.append(deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_mf, cov_theta_mf))
          #vi_pred_mean_fr.append(deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_fr, cov_theta_fr))
          vi_pred_var_mf.append(deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_mf, cov_theta_mf))
          #vi_pred_var_fr.append(deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_fr, cov_theta_fr))
         
        # Checking MF - MCMC vs deter
      
        plt.figure()
        plt.plot(X_40, y_40, 'ko')
        plt.plot(X_star_40, post_means_mf_40.T, 'grey', alpha=0.4)
        plt.plot(X_star_40, f_star_40, 'k')
        plt.plot(X_star_40, pp_mean_mf_40, color='coral')
        #plt.fill_between(X_star.ravel(), pp_mean_ml_40 -1.96*pp_std_ml_nf_40, pp_mean_ml_40 + 1.96*pp_std_ml_nf_40, color='coral', alpha=0.3)
        plt.plot(X_star_40, vi_pred_mean_mf, color='r')
        plt.fill_between(X_star.ravel(), lower_mf_40, upper_mf_40, color='coral', alpha=0.3)
        plt.fill_between(X_star_40.ravel(), (vi_pred_mean_mf - 1.96*np.sqrt(vi_pred_var_mf)).ravel(), (vi_pred_mean_mf + 1.96*np.sqrt(vi_pred_var_mf)).ravel(), alpha=0.5, color='r')

      
        # Checking FR - MCMC vs deter
      
        plt.figure()
        plt.plot(X_40, y_40, 'ko')
        plt.plot(X_star_40, post_means_fr_40.T, 'grey', alpha=0.4)
        plt.plot(X_star_40, f_star_40, 'k')
        plt.plot(X_star_40, pp_mean_fr_40, color='g')
        plt.plot(X_star_40, vi_pred_mean_fr, color='b')
        plt.fill_between(X_star_40.ravel(), lower_fr_40, upper_fr_40, color='g', alpha=0.3)
        plt.fill_between(X_star_40.ravel(), (vi_pred_mean_fr - 1.96*np.sqrt(vi_pred_var_fr)).ravel(), (vi_pred_mean_fr + 1.96*np.sqrt(vi_pred_var_fr)).ravel(), alpha=0.5)
         #plt.fill_between(X_star_40.ravel(), pp_mean_ml_40 - 1.96*pp_std_ml_nf_40, pp_mean_ml_40 + 1.96*pp_std_ml_nf_40, alpha=0.5)


