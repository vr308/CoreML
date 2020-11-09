wo#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:48:56 2019

@author: vidhi

"""

import pymc3 as pm
import theano.tensor as tt
from sampled import sampled
import pandas as pd
import numpy as np
from matplotlib import cm
import time as time
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import seaborn as sns
import csv
import scipy.stats as st
import warnings
import posterior_analysis as pa
import synthetic_data_generation as sdg
warnings.filterwarnings("ignore")

#----------------------------Loading persisted data----------------------------

def load_datasets(path, n_train, n_star):
      
       n_test = n_star - n_train
       X = np.asarray(pd.read_csv(path + 'X_' + str(n_train) + '.csv', header=None)).reshape(n_train,1)
       y = np.asarray(pd.read_csv(path + 'y_' + str(n_train) + '.csv', header=None)).reshape(n_train,)
       X_star = np.asarray(pd.read_csv(path + 'X_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,1)
       f_star = np.asarray(pd.read_csv(path + 'f_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,)
       return X, y, X_star, f_star

#----------------------------GP Inference-------------------------------------
    
# Type II ML 
    
def get_ml_report(X, y, X_star, f_star):
      
          kernel = Ck(50, (1e-10, 1e3)) * RBF(0.0001, length_scale_bounds=(1e-10, 1e3)) + WhiteKernel(1.0, noise_level_bounds=(1e-10,1000))
          
          gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
              
          # Fit to data 
          gpr.fit(X, y)        
          ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
          post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
          post_std = np.sqrt(np.diag(post_cov))
          post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
          post_samples = np.random.multivariate_normal(post_mean, post_cov , 10)
          rmse_ = pa.rmse(post_mean, f_star)
          lpd_ = -pa.log_predictive_density(f_star, post_mean, post_std)
          title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + '-LPD: ' + str(lpd_)     
          ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
                            'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
          return gpr, post_mean, post_std, post_std_nf, rmse_, lpd_, ml_deltas_dict, title
    
# For one training set size 
def get_ml_II_hyp_variance(X, y, X_star, f_star, runs):
      
          ml_deltas_runs = [];  rmse_runs = []; nlpd_runs = []; post_means = []; lml_runs = []
         
          kernel = Ck(50, (1e-3, 1e+7)) * RBF(1, length_scale_bounds=(1e-3, 1e+3)) + WhiteKernel(1.0, noise_level_bounds=(1e-3,1e+7))
          
          for i in np.arange(runs):
                print('Run ' + str(i))
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                gpr.fit(X, y)        
                ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
                post_mean, post_cov = gpr.predict(X_star, return_cov = True) # sklearn always predicts with noise
                post_std = np.sqrt(np.diag(post_cov))
                post_std_nf = np.sqrt(np.diag(post_cov) - ml_deltas[2])
                rmse_ = pa.rmse(post_mean, f_star)
                lpd_ = -pa.log_predictive_density(f_star, post_mean, post_std)
                lml = gpr.log_marginal_likelihood_value_
                
                ml_deltas_runs.append(ml_deltas)
                rmse_runs.append(rmse_)
                nlpd_runs.append(lpd_)
                lml_runs.append(lml)
                post_means.append(post_mean)
      
          return np.array(ml_deltas_runs).reshape(runs,3), rmse_runs, nlpd_runs, lml_runs, np.array(post_means)
              
def plot_hyp_variance(sig_sd_data, ls_data, noise_sd_data, labels, snr):
      
      plt.figure(figsize=(20,8))
      plt.subplot(131)
      sns.violinplot(data=sig_sd_data.T, orient='v', palette='Blues_d', cut=0)
      sns.swarmplot(data=sig_sd_data.T, color='.2')
      plt.xticks(np.arange(6),labels)
      plt.axhline(y=true_hyp[0], color='r', alpha=0.5, label='True value')
      plt.title('sig_sd', fontsize='small')
      plt.xlabel('N_train', fontsize='small')
      plt.legend(fontsize='small')
      plt.subplot(132)
      sns.violinplot(data=ls_data.T, orient='v', palette='Greens_d', cut=0)
      sns.swarmplot(data=ls_data.T, color='.2')
      plt.axhline(y=true_hyp[1], color='r', alpha=0.5,label='True value')
      plt.title('lengthscale', fontsize='small')
      plt.legend(fontsize='small')
      plt.xlabel('N_train', fontsize='small')
      plt.xticks(np.arange(6),labels)
      plt.subplot(133)
      sns.violinplot(data=noise_sd_data.T, orient='v', palette='Reds_d', inner='sticks')
      plt.xticks(np.arange(6),labels)
      plt.axhline(y=true_hyp[2], color='r', alpha=0.5,label='True value')
      plt.title('noise_sd', fontsize='small')
      plt.legend(fontsize='small')
      plt.xlabel('N_train', fontsize='small')
      plt.suptitle('Evolution / Variance of ML-II estimates vs. N' + 'snr: ' + str(snr), fontsize='small')

      
def plot_metric_curves(rmse_data, nlpd_data, labels):

      plt.figure(figsize=(15,8))
      plt.subplot(121)
      plt.violinplot(rmse_data, showmeans=True, bw_method=0.3)
      plt.xticks(np.arange(7)[1:], labels)
      plt.title('RMSE (Test)', fontsize='small')
      plt.xlabel('N_train')
      ax = plt.subplot(122)      
      vp = ax.violinplot(nlpd_data, showmeans=True, bw_method=0.3)
      for v in vp['bodies']:
            v.set_facecolor('red')
      for v in ('cbars','cmins','cmaxes','cmeans'):
            vp[v].set_edgecolor('red')
      plt.xticks(np.arange(7)[1:], labels)
      plt.title('Neg. Log Predictive Density (Test)', fontsize='small')
      plt.xlabel('N_train')


# Generative model for full Bayesian treatment

@sampled
def generative_model(X, y):
      
       # prior on lengthscale 
       log_ls = pm.Normal('log_ls',mu = 0, sd = 3)
       ls = pm.Deterministic('ls', tt.exp(log_ls))
       
        #prior on noise variance
       log_n = pm.Normal('log_n', mu = 0 , sd = 3)
       noise_sd = pm.Deterministic('noise_sd', tt.exp(log_n))
         
       #prior on signal variance
       log_s = pm.Normal('log_s', mu=0, sd = 3)
       sig_sd = pm.Deterministic('sig_sd', tt.exp(log_s))
       #sig_sd = pm.InverseGamma('sig_sd', 4, 4)
       
       # Specify the covariance function.
       cov_func = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
       
       #Prior
       trace_prior = pm.sample(draws=1000)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise_sd)
       

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
          #post_std_y = np.sqrt(np.diag(post_cov))
          return post_mean,  post_std

def analytical_gp2(y, K, K_s, K_ss, K_noise, K_inv, noise_var):
    
          L = np.linalg.cholesky(K_noise.eval())
          alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
          v = np.linalg.solve(L, K_s.eval())
          post_mean = np.dot(K_s.eval().T, alpha)
          post_cov = K_ss.eval() - v.T.dot(v)
          #post_std = np.sqrt(np.diag(post_cov))
          post_std_y = np.sqrt(np.diag(post_cov) + noise_var)
          return post_mean,  post_std_y
    
      
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
                         
#--------------Predictions------------------------------------
      
def compute_log_marginal_likelihood(K_noise, y):
      
      return np.log(st.multivariate_normal.pdf(y, cov=K_noise.eval()))
      
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
            #post_mean, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
            post_mean, post_std = analytical_gp2(y, K, K_s, K_ss, K_noise, K_inv, np.square(trace[i]['noise_sd']))
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

#-------------Plotting---------------------------------------

def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')
    

def plot_gp(X_star, f_star, X, y, post_mean, post_std, post_samples, title):
    
    plt.figure()
    if post_samples != []:
          plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
    plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3, alpha=0.5)
    plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), post_mean - 1.96*post_std, 
                     post_mean + 1.96*post_std, alpha=0.2, color='g',
                     label='95% CR')
    plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')
    
def plot_gp_ml_II_joint(X, y, X_star, pred_mean_ml, pred_std_ml, titles_ml, suptitle):
      
      plt.figure(figsize=(20,5))
      
      for i in [0,1,2,3]:
            plt.subplot(1,4,i+1)
            plt.plot(X_star[i], pred_mean_ml[i], color='r')
            plt.plot(X_star[i], f_star[i], 'k', linestyle='dashed')
            plt.plot(X[i], y[i], 'ko', markersize=2)
            plt.fill_between(X_star[i].ravel(), pred_mean_ml[i] -1.96*pred_std_ml[i], pred_mean_ml[i] + 1.96*pred_std_ml[i], color='r', alpha=0.3)
            plt.title(titles_ml[i], fontsize='x-small')
            plt.ylim(-30, 30)
      plt.suptitle('Type II ML '   + suptitle, fontsize='x-small') 
      plt.tight_layout(pad=1.5)

      
def plot_gp_hmc_joint(X, y, X_star, pred_mean_hmc, lower_hmc, upper_hmc, titles, suptitle):
      
      plt.figure(figsize=(20,5))
      
      for i in [0,1,2,3]:
            plt.subplot(1,4,i+1)
            plt.plot(X_star[i], pred_mean_hmc[i], color='b')
            plt.plot(X_star[i], f_star[i], 'k', linestyle='dashed')
            plt.plot(X[i], y[i], 'ko', markersize=2)
            plt.fill_between(X_star[i].ravel(), lower_hmc[i], upper_hmc[i], color='b', alpha=0.3)
            plt.title(titles[i], fontsize='x-small')
            plt.ylim(-30, 30)
      plt.suptitle('HMC ' + suptitle, fontsize='x-small')    
      plt.tight_layout(pad=1.5)
      
def plot_hmc_ml_joint(X, y, X_star, pred_mean_hmc, pred_mean_ml, lower_hmc, upper_hmc, pred_std_ml):
            
            plt.figure(figsize=(20,5))
            for i in [0,1,2,3]:
                  plt.subplot(1,4,i+1)
                  plt.plot(X_star[i], pred_mean_hmc[i], color='b', label='HMC')
                  plt.plot(X_star[i], pred_mean_ml[i], color='red', label='ML')
                  plt.plot(X_star[i], f_star[i], 'k', linestyle='dashed')
                  plt.plot(X[i], y[i], 'ko', markersize=2)
                  plt.fill_between(X_star[i].ravel(), lower_hmc[i], upper_hmc[i], color='b', alpha=0.3)
                  plt.fill_between(X_star[i].ravel(), pred_mean_ml[i] - 1.96*pred_std_ml_nf[i], pred_mean_ml[i] + 1.96*pred_std_ml_nf[i], color='r', alpha=0.2)
                  plt.ylim(-30, 30)
            plt.suptitle('HMC / ML' + suptitle, fontsize='x-small')    
            plt.tight_layout(pad=1.5)
            plt.legend(fontsize='x-small') 
      
def plot_hyp_convergence(tracks, n_train, true_hyp):
      
      plt.figure(figsize=(10,6))
      
      plt.subplot(131)
      plt.axhline(true_hyp[0], color='r', label=r'$\sigma_{f}^{2}$')
      plt.plot(n_train, tracks['sig_sd_track'], 'ro-')

      plt.subplot(132)
      plt.axhline(true_hyp[1], color='b', label=r'$\gamma$')
      plt.plot(n_train, tracks['ls_track'], 'bo-')

      plt.subplot(133)
      plt.plot(n_train, tracks['noise_sd_track'], 'go-')
      plt.axhline(true_hyp[2], color='g', label=r'$\sigma_{n}^{2}$')


#  PairGrid plot  - bivariate relationships
      
def pair_grid_plot(trace_df, ml_deltas, true_hyp_dict, color, title, varnames):

      g = sns.PairGrid(trace_df, vars=varnames, diag_sharey=False)
      g = g.map_lower(plot_bi_kde, ml_deltas=ml_deltas,true_hyp_dict=true_hyp_dict, color=color)
      g = g.map_diag(plot_hist, color=color)
      g = g.map_upper(plot_scatter, ml_deltas=ml_deltas, color=color)
      
      g.axes[0,0].axvline(ml_deltas[g.x_vars[0]], color='r')
      g.axes[1,1].axvline(ml_deltas[g.x_vars[1]], color='r')
      g.axes[2,2].axvline(ml_deltas[g.x_vars[2]], color='r')
      
      plt.suptitle(title, fontsize='small')

def plot_bi_kde(x,y, ml_deltas,true_hyp_dict, color, label):
      
      sns.kdeplot(x, y, n_levels=20, color=color, shade=True, shade_lowest=False, bw=0.5)
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      plt.axvline(true_hyp_dict[x.name], color='k', alpha=0.5)
      plt.axhline(true_hyp_dict[y.name], color='k', alpha=0.5)
      
def plot_hist(x, color, label):
      
      sns.distplot(x, bins=100, color=color, kde=False)

def plot_scatter(x, y, ml_deltas, color, label):
      
      plt.scatter(x, y, c=color, s=0.5, alpha=0.7)
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      
      
def plot_autocorrelation(trace_hmc, varnames, title):
      
      pm.autocorrplot(trace_hmc, varnames=varnames)
      plt.suptitle(title, fontsize='small')
      

# Marginal

def plot_simple_traceplot(trace, varnames, ml_deltas, true_hyp_dict, log, title):
      
      traces = pm.traceplot(trace, varnames=varnames, lines=ml_deltas, combined=True)
      
      for i in np.arange(3):
           
            delta = ml_deltas.get(str(varnames[i]))
            true = true_hyp_dict.get(str(varnames[i]))
            traces[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces[i][0].axvline(x=true, color='k', linestyle='dashed', alpha=0.8, label='True ' + str(np.round(true, 2)))
            traces[i][0].hist(trace[varnames[i]], bins=100, density=True, color='b', alpha=0.3)
            traces[i][1].axhline(y=delta, color='r', alpha=0.5) 
            traces[i][1].axhline(y=true, color='k', linestyle='dashed', alpha=0.9) 
            traces[i][0].legend(fontsize='x-small')
            if log:
                 traces[i][0].axes.set_xscale('log')
      plt.suptitle(title, fontsize='small')

def plot_spans_hmc(X, y, X_star, f_star, mean_spans_hmc, lower_hmc, upper_hmc, suffix_t):
      
      plt.figure(figsize=(20,5))
      
      for i in [0,1,2,3]:
            plt.subplot(1,4,i+1)
            plt.plot(X_star[i], mean_spans_hmc[i].T, color='dodgerblue', alpha=0.2)
            plt.plot(X_star[i], np.mean(mean_spans_hmc[i]), color='b')
            plt.plot(X_star[i], f_star[i], 'k', linestyle='dashed')
            plt.plot(X[i], y[i], 'ko', markersize=4)
            plt.fill_between(X_star[i].ravel(), lower_hmc[i], upper_hmc[i] , color='grey', alpha=0.2)
            plt.ylim(-30, 30)
            plt.xticks()
      plt.suptitle('HMC Span ' + suffix_t, fontsize='x-small')   
      plt.tight_layout(pad=1.5)
      
def traceplot_compare(model, trace_hmc, trace_mf, trace_fr, varnames, deltas):

      traces = pm.traceplot(trace_hmc, varnames, lines=deltas, combined=True)
      
      rv_mapping = {'sig_sd': model.log_s_interval__, 'ls': model.log_ls_interval__, 'noise_sd': model.log_n_interval__}

      means_mf = mf.approx.bij.rmap(mf.approx.mean.eval())  
      std_mf = mf.approx.bij.rmap(mf.approx.std.eval())  
      
      means_fr = fr.approx.bij.rmap(fr.approx.mean.eval())  
      std_fr = fr.approx.bij.rmap(fr.approx.std.eval())  
      
      for i in np.arange(3):
            
            delta = deltas.get(str(varnames[i]))
            xmax = max(max(trace_hmc[varnames[i]]), delta)
            xmin = min(min(trace_hmc[varnames[i]]), delta)
            range_i = np.linspace(xmin, xmax, 1000)  
            traces[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces[i][0].hist(trace_hmc[varnames[i]], bins=100, density=True, color='b', alpha=0.3)
            traces[i][1].axhline(y=delta, color='r', alpha=0.5)
            traces[i][0].plot(range_i, get_implicit_variational_posterior(rv_mapping.get(varnames[i]), means_fr, std_fr, range_i), color='green')
            traces[i][0].plot(range_i, get_implicit_variational_posterior(rv_mapping.get(varnames[i]), means_mf, std_mf, range_i), color='coral')
            #traces_part1[i][0].axes.set_ylim(0, 0.005)
            traces[i][0].legend(fontsize='x-small')
            
# Variational approximation
            
def get_implicit_variational_posterior(var, means, std, x):
      
      sigmoid = lambda x : 1 / (1 + np.exp(-x))

      eps = lambda x : var.distribution.transform_used.forward_val(np.log(x))
      backward_theta = lambda x: var.distribution.transform_used.backward(x).eval()   
      width = (var.distribution.transform_used.b -  var.distribution.transform_used.a).eval()
      total_jacobian = lambda x: x*(width)*sigmoid(eps(x))*(1-sigmoid(eps(x)))
      pdf = lambda x: st.norm.pdf(eps(x), means[var.name], std[var.name])/total_jacobian(x)
      return pdf(x)

      
#-----------------Trace post-processing & analysis ------------------------------------

def get_trace_means(trace, varnames):
      
      trace_means = []
      for i in varnames:
            trace_means.append(trace[i].mean())
      return trace_means

def get_trace_sd(trace, varnames):
      
      trace_sd = []
      for i in varnames:
            trace_sd.append(trace[i].std())
      return trace_sd



def trace_report(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, ml_deltas, true_hyp):
      
      hyp_mean_mf = np.round(get_trace_means(trace_mf, varnames),3)
      hyp_sd_mf = np.round(get_trace_sd(trace_mf, varnames),3)
      
      hyp_mean_fr = np.round(get_trace_means(trace_fr, varnames),3)
      hyp_sd_fr = np.round(get_trace_sd(trace_fr, varnames),3)
      
      means_mf = mf.approx.bij.rmap(mf.approx.mean.eval())  
      std_mf = mf.approx.bij.rmap(mf.approx.std.eval())  
      
      means_fr = fr.approx.bij.rmap(fr.approx.mean.eval())  
      std_fr = fr.approx.bij.rmap(fr.approx.std.eval())  

      traces = pm.traceplot(trace_hmc, varnames=varnames, lines=ml_deltas, combined=True, bw=1)
      
      l_int = np.linspace(min(trace_fr['ls'])-1, max(trace_fr['ls'])+1, 1000)
      n_int = np.linspace(min(trace_fr['noise_sd'])-1, max(trace_fr['noise_sd'])+1, 1000)
      s_int = np.linspace(min(trace_fr['sig_sd'])-1, max(trace_fr['sig_sd'])+1, 1000)
      
      mf_rv = [mf.approx.model.log_s_interval__, mf.approx.model.log_ls_interval__, mf.approx.model.log_n_interval__]
      fr_rv = [fr.approx.model.log_s_interval__, fr.approx.model.log_ls_interval__, fr.approx.model.log_n_interval__]
      
      ranges = [s_int, l_int, n_int]
      
      for i, j in [(0,0), (1,0), (2,0)]:
            
            traces[i][j].axvline(x=hyp_mean_mf[i], color='coral', alpha=0.5, label='MF ' + str(hyp_mean_mf[i]))
            traces[i][j].axvline(x=hyp_mean_fr[i], color='g', alpha=0.5, label='FR ' + str(hyp_mean_fr[i]))
            traces[i][j].axvline(x=ml_deltas[varnames[i]], color='r', alpha=0.5, label='ML ' + str(ml_deltas[varnames[i]]))
            traces[i][j].axvline(x=true_hyp[i], color='k', linestyle='--', label='True ' + str(true_hyp[i]))
            #traces[i][j].axes.set_xscale('log')
            
            traces[i][j].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
            #traces[i][j].hist(trace_mf[varnames[i]], bins=100, normed=True, color='coral', alpha=0.3)
            #traces[i][j].hist(trace_fr[varnames[i]], bins=100, normed=True, color='g', alpha=0.3)

            traces[i][j].plot(ranges[i], get_implicit_variational_posterior(mf_rv[i], means_mf, std_mf, ranges[i]), color='coral')
            traces[i][j].plot(ranges[i], get_implicit_variational_posterior(fr_rv[i], means_fr, std_fr, ranges[i]), color='g')
            traces[i][j].legend(fontsize='x-small')
            
def plot_lml_surface_3way(gpr, sig_sd, lengthscale, noise_sd):
    
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    l_log = np.logspace(-5, 5, 100)
    noise_log  = np.logspace(-6, 6, 100)
    l_log_mesh, noise_log_mesh = np.meshgrid(l_log, noise_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(sig_sd), l_log_mesh[i, j], np.square(noise_log_mesh[i, j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, noise_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, noise_sd, 'rx')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise sd")
    
    plt.subplot(132)
    l_log = np.logspace(-5, 4, 100)
    signal_log  = np.logspace(-5, 4, 100)
    l_log_mesh, signal_log_mesh = np.meshgrid(l_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), l_log_mesh[i, j], np.square(noise_sd)]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=3)
    plt.contourf(l_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Signal sd")
    
    plt.subplot(133)
    noise_log = np.logspace(-5, 2, 100)
    signal_log  = np.logspace(-5, 2, 100)
    noise_log_mesh, signal_log_mesh = np.meshgrid(noise_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([np.square(signal_log_mesh[i,j]), lengthscale, np.square(noise_log_mesh[i,j])]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=3)
    plt.contourf(noise_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(noise_sd, sig_sd, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise sd")
    plt.ylabel("Signal sd")    
    plt.suptitle('LML Surface ' + '\n' + str(gpr.kernel_), fontsize='small')
            
      
if __name__ == "__main__":

      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/'
      home_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/1d/'
      
      path = uni_path

      # Edit here to change generative model
      
      input_dist =  'Unif'
      snr = 6
      n_train = [10, 20, 40, 80, 100, 120]  
      suffix = input_dist + '/' + 'snr_' + str(snr) + '/'
      true_hyp = [np.round(np.sqrt(625),3), 5, np.round(np.sqrt(100),3)]
      
      suffix_t = 'snr = ' + str(snr) + ', ' + input_dist

      data_path = path + suffix
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/1d/' + suffix 
      
      true_hyp_dict = {'sig_sd': true_hyp[0], 'ls': true_hyp[1] , 'noise_sd': true_hyp[2], 'log_s': np.log(true_hyp[0]),  'log_ls':np.log(true_hyp[1]) , 'log_n':np.log(true_hyp[2])}
      varnames = ['sig_sd', 'ls', 'noise_sd']
      log_varnames = ['log_s', 'log_ls', 'log_n']
      
      #----------------------------------------------------
      # Joint Analysis
      #----------------------------------------------------
      
      # Type II ML
      n_star = 500
      
      X_10, y_10, X_star_10, f_star_10 = load_datasets(data_path, 10, n_star)
      X_20, y_20, X_star_20, f_star_20 = load_datasets(data_path, 20, n_star)
      X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40, n_star)
      X_80, y_80, X_star_80, f_star_80 = load_datasets(data_path, 80, n_star)
      X_100, y_100, X_star_100, f_star_100 = load_datasets(data_path, 100, n_star)
      X_120, y_120, X_star_120, f_star_120 = load_datasets(data_path, 120, n_star)

      
      X = [X_10, X_20, X_40, X_80, X_100, X_120]
      y = [y_10, y_20, y_40, y_80, y_100, y_120]
      X_star = [X_star_10, X_star_20, X_star_40, X_star_80, X_star_120]
      f_star = [f_star_10, f_star_20, f_star_40, f_star_80, f_star_120] 
      
      # Plot metric curves RMSE, NLPD for ML-II

      hyp_10, rmse_10, nlpd_10, lml_10, means_10 = get_ml_II_hyp_variance(X_10, y_10, X_star_10, f_star_10, runs)  
      hyp_20, rmse_20, nlpd_20, lml_20, means_20 = get_ml_II_hyp_variance(X_20, y_20, X_star_20, f_star_20, runs)  
      hyp_40, rmse_40, nlpd_40, lml_40, means_40 = get_ml_II_hyp_variance(X_40, y_40, X_star_40, f_star_40, runs)  
      hyp_80, rmse_80, nlpd_80, lml_80, means_80 = get_ml_II_hyp_variance(X_80, y_80, X_star_80, f_star_80, 2)  
      hyp_100, rmse_100, nlpd_100, lml_100, means_80 = get_ml_II_hyp_variance(X_100, y_100, X_star_100, f_star_100, runs)  
      hyp_120, rmse_120, nlpd_120, lml_120, means_120 = get_ml_II_hyp_variance(X_120, y_120, X_star_120, f_star_120, runs)  
      
      rmse_data = np.vstack((rmse_10, rmse_20, rmse_40, rmse_80, rmse_100, rmse_120)).T
      nlpd_data = np.vstack((nlpd_10, nlpd_20, nlpd_40, nlpd_80, nlpd_100, nlpd_120)).T
      labels=n_train
      
      # Function Call
      plot_metric_curves(rmse_data, nlpd_data, labels)
      
     # Plot hyp variance vs. N for ML-II
      
      sig_sd_data = np.vstack((np.sqrt(hyp_10[:,0]), np.sqrt(hyp_20[:,0]), np.sqrt(hyp_40[:,0]), np.sqrt(hyp_80[:,0]), np.sqrt(hyp_100[:,0]), np.sqrt(hyp_120[:,0])))
      ls_data = np.vstack((hyp_10[:,1], hyp_20[:,1], hyp_40[:,1], hyp_80[:,1], hyp_100[:,1], hyp_120[:,1]))
      noise_sd_data = np.vstack((np.sqrt(hyp_10[:,2]), np.sqrt(hyp_20[:,2]), np.sqrt(hyp_40[:,2]), np.sqrt(hyp_80[:,2]), np.sqrt(hyp_100[:,2]), np.sqrt(hyp_120[:,2])))
      
      # Function Call
      plot_hyp_variance(sig_sd_data, ls_data, noise_sd_data, labels, 6)
      
      
      # Collecting ML stats for 1 generative model

      gpr_10, pp_mean_ml_10, pp_std_ml_10, pp_std_ml_nf_10, rmse_ml_10, lpd_ml_10, ml_deltas_dict_10, title_10 =  get_ml_report(X_10, y_10, X_star_10, f_star_10)
      gpr_20, pp_mean_ml_20, pp_std_ml_20, pp_std_ml_nf_20, rmse_ml_20, lpd_ml_20, ml_deltas_dict_20, title_20 =  get_ml_report(X_20, y_20, X_star_20, f_star_20)
      gpr_40, pp_mean_ml_40, pp_std_ml_40, pp_std_ml_nf_40, rmse_ml_40, lpd_ml_40, ml_deltas_dict_40, title_40 =  get_ml_report(X_40, y_40, X_star_40, f_star_40)
      gpr_80, pp_mean_ml_80, pp_std_ml_80, pp_std_ml_nf_80, rmse_ml_80, lpd_ml_80, ml_deltas_dict_80, title_80 = get_ml_report(X_80, y_80, X_star_80, f_star_80)  
      gpr_100, pp_mean_ml_100, pp_std_ml_100, pp_std_ml_nf_100, rmse_ml_100, lpd_ml_100, ml_deltas_dict_100, title_100 =  get_ml_report(X_100, y_100, X_star_100, f_star_100)  
      gpr_120, pp_mean_ml_120, pp_std_ml_120, pp_std_ml_nf_120, rmse_ml_120, lpd_ml_120, ml_deltas_dict_120, title_120 =  get_ml_report(X_120, y_120, X_star_120, f_star_120)
     
      # LML Surface
      
      plot_lml_surface_3way(gpr_5, ml_deltas_dict_5['sig_sd'], ml_deltas_dict_5['ls'], ml_deltas_dict_5['noise_sd'])
      plot_lml_surface_3way(gpr_10, ml_deltas_dict_10['sig_sd'], ml_deltas_dict_10['ls'], ml_deltas_dict_10['noise_sd'])
      plot_lml_surface_3way(gpr_20, ml_deltas_dict_20['sig_sd'], ml_deltas_dict_20['ls'], ml_deltas_dict_20['noise_sd'])
      plot_lml_surface_3way(gpr_40, ml_deltas_dict_40['sig_sd'], ml_deltas_dict_40['ls'], ml_deltas_dict_40['noise_sd'])

      # Collecting means and stds in a list for plotting

      titles_ml = [ title_10, title_20, title_40]
      pred_mean_ml = [pp_mean_ml_5, pp_mean_ml_10, pp_mean_ml_20, pp_mean_ml_40] 
      pred_std_ml = [pp_std_ml_5, pp_std_ml_10, pp_std_ml_20, pp_std_ml_40]
      pred_std_ml_nf = [pp_std_ml_nf_5, pp_std_ml_nf_10, pp_std_ml_nf_20, pp_std_ml_nf_40]
      rmse_track_ml = [rmse_ml_5, rmse_ml_10, rmse_ml_20, rmse_ml_40]
      lpd_track_ml = [lpd_ml_5, lpd_ml_10, lpd_ml_20, lpd_ml_40]
      
      # Collecting hyp tracks for plotting
      
      sig_sd_track = [ml_deltas_dict_5['sig_sd'], ml_deltas_dict_10['sig_sd'], ml_deltas_dict_20['sig_sd'], ml_deltas_dict_40['sig_sd']]
      noise_sd_track = [ml_deltas_dict_5['noise_sd'], ml_deltas_dict_10['noise_sd'], ml_deltas_dict_20['noise_sd'], ml_deltas_dict_40['noise_sd']]
      ls_track = [ml_deltas_dict_5['ls'], ml_deltas_dict_10['ls'], ml_deltas_dict_20['ls'], ml_deltas_dict_40['ls']]
      
      tracks = {}
      tracks.update({'sig_sd_track': sig_sd_track})
      tracks.update({'noise_sd_track': noise_sd_track})
      tracks.update({'ls_track': ls_track})
            
      # Convergence to hyp 
      
      plot_hyp_convergence(tracks, n_train, true_hyp)

      # ML Report
      
      plot_gp_ml_II_joint(X, y, X_star, pred_mean_ml, pred_std_ml_nf, titles_ml, suffix_t)
       

      #-----------------------------Full Bayesian HMC--------------------------------------- 
            
      # Generating traces with NUTS
      
      with generative_model(X=X_10, y=y_10): trace_hmc_10 = pm.sample(draws=1000, tune=1000)
      with generative_model(X=X_20, y=y_20): trace_hmc_20 = pm.sample(draws=1000, tune=1000)
      with generative_model(X=X_40, y=y_40): trace_hmc_40 = pm.sample(draws=1000, tune=1000)
      with generative_model(X=X_80, y=y_80): trace_hmc_80 = pm.sample(draws=1000, tune=1000)
      with generative_model(X=X_100, y=y_100): trace_hmc_100 = pm.sample(draws=1000, tune=1000)
      with generative_model(X=X_120, y=y_120): trace_hmc_120 = pm.sample(draws=1000, tune=1000)

      # Persist traces
       
      with generative_model(X=X_5, y=y_5): pm.save_trace(trace_hmc_5, directory = results_path + 'traces_hmc/x_5/')
      with generative_model(X=X_10, y=y_10): pm.save_trace(trace_hmc_10, directory = results_path + 'traces_hmc/x_10/')
      with generative_model(X=X_20, y=y_20):  pm.save_trace(trace_hmc_20, directory = results_path + 'traces_hmc/x_20/')
      with generative_model(X=X_40, y=y_40):  pm.save_trace(trace_hmc_40, directory = results_path + 'traces_hmc/x_40/')
      
      trace_hmc_5 = pm.load_trace(results_path + 'traces_hmc/x_5/', model=generative_model(X=X_5, y=y_5))
      trace_hmc_10 = pm.load_trace(results_path + 'traces_hmc/x_10/', model=generative_model(X=X_10, y=y_10))
      trace_hmc_20 = pm.load_trace(results_path + 'traces_hmc/x_20/', model=generative_model(X=X_20, y=y_20))
      trace_hmc_40 = pm.load_trace(results_path + 'traces_hmc/x_40/', model=generative_model(X=X_40, y=y_40))

      trace_hmc_5_df = pm.trace_to_dataframe(trace_hmc_5)
      trace_hmc_10_df = pm.trace_to_dataframe(trace_hmc_10)
      trace_hmc_20_df = pm.trace_to_dataframe(trace_hmc_20)
      trace_hmc_40_df = pm.trace_to_dataframe(trace_hmc_40)

      # Traceplot Marginals
       
       title_b_5 = input_dist +  ',  SNR: ' + str(snr) + ', N: ' + str(5)
       title_b_10 =  input_dist +  ',  SNR: ' + str(snr) + ', N: ' + str(10)
       title_b_20 = input_dist +  ',  SNR: ' + str(snr) + ', N: ' + str(20)
       title_b_40 = input_dist +  ',  SNR: ' + str(snr) + ', N: ' + str(40)
       
       plot_simple_traceplot(trace_hmc_5, varnames, ml_deltas_dict_5, true_hyp_dict, log=False, title = title_b_5)
       plot_simple_traceplot(trace_hmc_10, varnames, ml_deltas_dict_10, true_hyp_dict, log=False, title = title_b_10)
       plot_simple_traceplot(trace_hmc_20, varnames, ml_deltas_dict_20,true_hyp_dict, log=False, title = title_b_20)
       plot_simple_traceplot(trace_hmc_40, varnames, ml_deltas_dict_40, true_hyp_dict, log=True, title = title_b_40)
       
       # Autocorrelation plots
       
       plot_autocorrelation(trace_hmc_5, varnames, title_b_5)
       plot_autocorrelation(trace_hmc_10, varnames, title_b_10)
       plot_autocorrelation(trace_hmc_20, varnames, title_b_20)
       plot_autocorrelation(trace_hmc_40, varnames, title_b_40)
       
       # Sampler stats
       
       summary_df_5 = pm.summary(trace_hmc_5)
       summary_df_10 = pm.summary(trace_hmc_10) 
       summary_df_20 = pm.summary(trace_hmc_20)
       summary_df_40 = pm.summary(trace_hmc_40)
       
       summary_df_5.to_csv(results_path + 'summary_hmc_5.csv', sep=',')
       summary_df_10.to_csv(results_path + 'summary_hmc_10.csv', sep=',')
       summary_df_20.to_csv(results_path + 'summary_hmc_20.csv', sep=',')
       summary_df_40.to_csv(results_path + 'summary_hmc_40.csv', sep=',')

       # Pair-Grid report
      
       pair_grid_plot(trace_hmc_5_df, ml_deltas_dict_5,  true_hyp_dict, 'b', title_b_5, varnames)
       pair_grid_plot(trace_hmc_10_df, ml_deltas_dict_10, true_hyp_dict, 'b', title_b_10, varnames)
       pair_grid_plot(trace_hmc_20_df, ml_deltas_dict_20, true_hyp_dict, 'b', title_b_20, varnames)
       pair_grid_plot(trace_hmc_40_df, ml_deltas_dict_40, true_hyp_dict, 'b', title_b_40, varnames)
       
       pair_grid_plot(trace_hmc_5_df, ml_deltas_dict_5, true_hyp_dict, 'b', title_b_5, log_varnames)
       pair_grid_plot(trace_hmc_10_df, ml_deltas_dict_10,true_hyp_dict, 'b', title_b_10, log_varnames)
       pair_grid_plot(trace_hmc_20_df, ml_deltas_dict_20, true_hyp_dict, 'b', title_b_20, log_varnames)
       pair_grid_plot(trace_hmc_40_df, ml_deltas_dict_40, true_hyp_dict, 'b', title_b_40, log_varnames)

       # Predictive means and stds - generate them 
       
       write_posterior_predictive_samples(trace_hmc_10, 40, X_10, y_10,  X_star_10, results_path, 'hmc')
       write_posterior_predictive_samples(trace_hmc_20, 40, X_20, y_20,  X_star_20, results_path, 'hmc')
       write_posterior_predictive_samples(trace_hmc_40, 40, X_40, y_40, X_star_40, results_path, 'hmc') 
       write_posterior_predictive_samples(trace_hmc_80, 50, X_80, y_80, X_star_80, results_path, 'hmc')
       write_posterior_predictive_samples(trace_hmc_100, 50, X_100, y_100, X_star_100, results_path, 'hmc')
       write_posterior_predictive_samples(trace_hmc_120, 50, X_120, y_120, X_star_120, results_path, 'hmc')


       post_means_hmc_10, post_stds_hmc_10 = load_post_samples(results_path + 'means_hmc_10.csv', results_path + 'std_hmc_10.csv')
       post_means_hmc_20, post_stds_hmc_20 = load_post_samples(results_path + 'means_hmc_20.csv', results_path + 'std_hmc_20.csv')
       post_means_hmc_40, post_stds_hmc_40 = load_post_samples(results_path + 'means_hmc_40.csv', results_path + 'std_hmc_40.csv')
       post_means_hmc_80, post_stds_hmc_80 = load_post_samples(results_path + 'means_hmc_80.csv', results_path + 'std_hmc_80.csv')  
       post_means_hmc_100, post_stds_hmc_100 = load_post_samples(results_path + 'means_hmc_100.csv', results_path + 'std_hmc_100.csv')
       post_means_hmc_120, post_stds_hmc_120 = load_post_samples(results_path + 'means_hmc_120.csv', results_path + 'std_hmc_120.csv')
       
       trace_hmc_5 = pd.read_csv(results_path + 'trace_hmc_5.csv')
       trace_hmc_10 = pd.read_csv(results_path + 'trace_hmc_10.csv')
       trace_hmc_20 = pd.read_csv(results_path + 'trace_hmc_20.csv')
       trace_hmc_40 = pd.read_csv(results_path + 'trace_hmc_40.csv')
       
       def get_rmse_post_samples(post_means_hmc, post_stds_hmc, f_star):
             
             rmse = []
             nlpd= []
             for i in np.arange(len(post_means_hmc)):
                   print('Computing RMSE / NLPD ' + str(i))
                   rmse.append(pa.rmse(post_means_hmc.ix[i], f_star))
                   nlpd.append(-pa.log_predictive_density(f_star, post_means_hmc.ix[i], post_stds_hmc.ix[i]))
             return rmse, nlpd
                   
                   
      rmse_10, nlpd_10 = get_rmse_post_samples(post_means_hmc_10, post_stds_hmc_10, f_star_10)   
      rmse_20, nlpd_20 = get_rmse_post_samples(post_means_hmc_20, post_stds_hmc_20, f_star_20)    
      rmse_40, nlpd_40 = get_rmse_post_samples(post_means_hmc_40, post_stds_hmc_40, f_star_40) 
      rmse_80, nlpd_80 = get_rmse_post_samples(post_means_hmc_80, post_stds_hmc_80, f_star_80)    
      rmse_100, nlpd_100 = get_rmse_post_samples(post_means_hmc_100, post_stds_hmc_100, f_star_100)    
      rmse_120, nlpd_120 = get_rmse_post_samples(post_means_hmc_120, post_stds_hmc_120, f_star_120)    
      
      rmse_hmc_data = np.vstack((rmse_10[0:20], rmse_20[0:20], rmse_40[0:20], rmse_80[0:20], rmse_100[0:20], rmse_120[0:20])).T
      nlpd_hmc_data = np.vstack((nlpd_10[0:20], nlpd_20[0:20], nlpd_40[0:20], nlpd_80[0:20], nlpd_100[0:20], nlpd_120[0:20])).T
      
      plot_metric_curves(rmse_hmc_data, nlpd_hmc_data, labels)

      plt.figure()
      plt.subplot(121)
      plt.violinplot(rmse_10,  vert=True)
      plt.subplot(122)
      
      # Final predictive mean and stds

       # N = 5 --------------------------------------------------------------------------------------------
       
       u5 = np.ones(len(post_means_hmc_5))
       
       pp_mean_hmc_5 = get_posterior_predictive_mean(post_means_hmc_5, weights=None)
       
       lower_hmc_5, upper_hmc_5 = get_posterior_predictive_uncertainty_intervals(post_means_hmc_5, post_stds_hmc_5, u5)
       
       rmse_hmc_5 = rmse(pp_mean_hmc_5, f_star_5)
       
       lppd_hmc_5, lpd_hmc_5 = log_predictive_mixture_density(f_star_5, post_means_hmc_5, post_stds_hmc_5, None)

       title_hmc_5 = 'RMSE: ' + str(rmse_hmc_5) + '\n' + '-LPD: ' + str(-lpd_hmc_5)
       
       # N = 10 --------------------------------------------------------------------------------------------
       
       u10 = np.ones(len(post_means_hmc_10))

       pp_mean_hmc_10 = get_posterior_predictive_mean(post_means_hmc_10, weights=None)
       
       lower_hmc_10, upper_hmc_10 = get_posterior_predictive_uncertainty_intervals(post_means_hmc_10, post_stds_hmc_10, u10)                 
       
       rmse_hmc_10 = rmse(pp_mean_hmc_10, f_star_10)

       lppd_hmc_10, lpd_hmc_10 = log_predictive_mixture_density(f_star_10, post_means_hmc_10, post_stds_hmc_10, None)

       title_hmc_10 = 'RMSE: ' + str(rmse_hmc_10) + '\n' + '-LPD: ' + str(-lpd_hmc_10)
      
       # N = 20 ----------------------------------------------------------------------------------------------
       
       u20 = np.ones(len(post_means_hmc_20))

       pp_mean_hmc_20 = get_posterior_predictive_mean(post_means_hmc_20, weights=None)
       
       lower_hmc_20, upper_hmc_20 = get_posterior_predictive_uncertainty_intervals(post_means_hmc_20, post_stds_hmc_20, u20)         
       
       rmse_hmc_20 = rmse(pp_mean_hmc_20, f_star_20)

       lppd_hmc_20, lpd_hmc_20 = log_predictive_mixture_density(f_star_20, post_means_hmc_20, post_stds_hmc_20, None)

       title_hmc_20 = 'RMSE: ' + str(rmse_hmc_20) + '\n' + '-LPD: ' + str(-lpd_hmc_20)
       
      # N = 40 --------------------------------------------------------------------------------------------------
      
       u40 = np.ones(len(post_means_hmc_40))

       pp_mean_hmc_40 = get_posterior_predictive_mean(post_means_hmc_40, weights=None)
       
       lower_hmc_40, upper_hmc_40 = get_posterior_predictive_uncertainty_intervals(post_means_hmc_40, post_stds_hmc_40, u40)
       
       rmse_hmc_40 = rmse(pp_mean_hmc_40, f_star_40)

       lppd_hmc_40, lpd_hmc_40 = log_predictive_mixture_density(f_star_40, post_means_hmc_40, post_stds_hmc_40, None)

       title_hmc_40 = 'RMSE: ' + str(rmse_hmc_40) + '\n' + '-LPD: ' + str(-lpd_hmc_40)
       
      # Collecting means and stds in a list for plotting

      pred_mean_hmc = [pp_mean_hmc_5, pp_mean_hmc_10, pp_mean_hmc_20, pp_mean_hmc_40] 
      #pred_mean_w_hmc = [pp_mean_hmc_w_5, pp_mean_hmc_w_10, pp_mean_hmc_w_20, pp_mean_hmc_w_40] 
      
      lower_hmc = [lower_hmc_5,lower_hmc_10, lower_hmc_20, lower_hmc_40]
      upper_hmc = [upper_hmc_5, upper_hmc_10, upper_hmc_20, upper_hmc_40]
      
      rmse_track_hmc = [rmse_hmc_5, rmse_hmc_10, rmse_hmc_20, rmse_hmc_40]
      
      lpd_track_hmc = [-lpd_hmc_5, -lpd_hmc_10, -lpd_hmc_20, -lpd_hmc_40]
      
      titles_hmc = [title_hmc_5, title_hmc_10, title_hmc_20, title_hmc_40]

      mean_spans_hmc = [post_means_hmc_5, post_means_hmc_10, post_means_hmc_20, post_means_hmc_40]
      std_spans_hmc = [post_stds_hmc_5, post_stds_hmc_10, post_stds_hmc_20, post_stds_hmc_40]

      # HMC Report  

      plot_gp_hmc_joint(X, y, X_star, pred_mean_hmc, lower_hmc, upper_hmc, titles_hmc, suffix_t)
      #plot_gp_hmc_joint(X, y, X_star, pred_mean_w_hmc, lower_w_hmc, upper_w_hmc, titles_w_hmc, suffix_t)

      # Plot Spans
      
      plot_spans_hmc(X, y, X_star, f_star, mean_spans_hmc, lower_hmc, upper_hmc, suffix_t)
            
      # Plot HMC weighted and unweighted predictions
      
      #plot_hmc_weighted_unweighted(X, y, X_star, pred_mean_hmc, pred_mean_w_hmc, lower_hmc, lower_w_hmc, upper_hmc, upper_w_hmc, suffix_t)
      
      # Plot HMC weighted w ML-II
      
      plot_hmc_ml_joint(X, y, X_star, pred_mean_hmc, pred_mean_ml, lower_hmc, upper_hmc, pred_std_ml_nf)
      #plot_hmc_ml_joint(X, y, X_star, pred_mean_w_hmc, pred_mean_ml, lower_w_hmc, upper_w_hmc, pred_std_ml_nf)

      # Metrics 
      plt.figure(figsize=(10,5))
      
      plt.subplot(121)
      # set width of bar
      barWidth = 0.25
       
      # Set position of bar on X axis
      r1 = np.arange(4)
      r2 = [x + barWidth for x in r1]
      #r3 = [x + barWidth for x in r2]
       
      # Make the plot
      
      plt.bar(r1, rmse_track_ml, color='r', width=barWidth, edgecolor='white', label='ML', alpha=0.4)
      plt.bar(r2, rmse_track_hmc, color='b', width=barWidth, edgecolor='white', label='HMC', alpha=0.5)
      #plt.bar(r3, rmse_track_w_hmc, color='purple', width=barWidth, edgecolor='white', label='HMC-W', alpha=0.5)
       
      # Add xticks on the middle of the group bars
      plt.xlabel('N Train', fontweight='bold')
      plt.xticks([r + barWidth for r in range(4)], n_train)
      plt.title('RMSE  ' + suffix_t, fontsize='x-small')
      # Create legend & Show graphic
      plt.legend()
      
      plt.subplot(122)
      
      barWidth = 0.25
      
      # Set position of bar on X axis
      r1 = np.arange(4)
      r2 = [x + barWidth for x in r1]
      #r3 = [x + barWidth for x in r2]
       
      # Make the plot
      plt.bar(r1, lpd_track_ml, color='r', width=barWidth, edgecolor='white', label='ML', alpha=0.4)
      plt.bar(r2, lpd_track_hmc, color='b', width=barWidth, edgecolor='white', label='HMC', alpha=0.5)
      #plt.bar(r3, rmse_track_w_hmc, color='purple', width=barWidth, edgecolor='white', label='HMC-W', alpha=0.5)
       
      # Add xticks on the middle of the group bars
      plt.xlabel('N Train', fontweight='bold')
      plt.xticks([r + barWidth for r in range(4)], n_train)
      plt.title('-LPD  ' + suffix_t, fontsize='x-small')
      # Create legend & Show graphic
      plt.legend()

      
      




      
      
      

