#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:36:52 2019

@author: vidhi
"""

import matplotlib.pylab as plt
import theano.tensor as tt
import pymc3 as pm
from theano.tensor.nlinalg import matrix_inverse
import pandas as pd
from sampled import sampled
import csv
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import seaborn as sns
import warnings
from autograd import grad, elementwise_grad, jacobian, hessian
import autograd.numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")

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
          title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '  ' + 'NLPD: ' + str(lpd_)     
          ml_deltas_dict = {'ls': ml_deltas[1], 'noise_sd': np.sqrt(ml_deltas[2]), 'sig_sd': np.sqrt(ml_deltas[0]), 
                            'log_ls': np.log(ml_deltas[1]), 'log_n': np.log(np.sqrt(ml_deltas[2])), 'log_s': np.log(np.sqrt(ml_deltas[0]))}
          return gpr, post_mean, post_std, post_std_nf, rmse_, lpd_, ml_deltas_dict, title
    
def get_star_kernel_matrix_blocks(X, X_star, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          return K_s.eval(), K_ss.eval()

def get_kernel_matrix_blocks(X, X_star, n_train, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K = cov(X)
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          K_noise = K + np.square(point['noise_sd'])*tt.eye(n_train)
          K_inv = matrix_inverse(K_noise)
          return K.eval(), K_s.eval(), K_ss.eval(), K_noise.eval(), K_inv.eval()

def analytical_gp(y, K, K_s, K_ss, K_noise, K_inv):
    
          L = np.linalg.cholesky(K_noise)
          alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
          v = np.linalg.solve(L, K_s)
          post_mean = np.dot(K_s.T, alpha)
          post_cov = K_ss - v.T.dot(v)
          post_std = np.sqrt(np.diag(post_cov))
          return post_mean,  post_std
    
def compute_log_marginal_likelihood(K_noise, y):
      
      return np.log(st.multivariate_normal.pdf(y, cov=K_noise))

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
       
def update_param_dict(model, param_dict, summary_trace, name_mapping):
      
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
             
def transform_tracker_values(tracker, param_dict, raw_mapping, name_mapping):

      mean_df = pd.DataFrame(np.array(tracker['mean']), columns=list(param_dict['mu'].keys()))
      sd_df = pd.DataFrame(np.array(tracker['std']), columns=list(param_dict['mu'].keys()))
      for i in mean_df.columns:
            print(i)
            if (i[-2:] == '__'):
                 mean_df[name_mapping[i]] = np.exp(raw_mapping.get(i).distribution.transform_used.backward(mean_df[i]).eval()) 
      return mean_df, sd_df

def convergence_report(tracker_mf, tracker_fr, mf_param, fr_param, true_hyp, varnames, rev_name_mapping, raw_mapping, name_mapping):
      
      # Plot Negative ElBO track with params in true space
      
      mean_mf_df, sd_mf_df = transform_tracker_values(tracker_mf, mf_param, raw_mapping, name_mapping)
      mean_fr_df, sd_fr_df = transform_tracker_values(tracker_fr, fr_param, raw_mapping, name_mapping)

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
      
def trace_report(mf, fr, means_mf, std_mf, means_fr, std_fr, true_hyp, trace_mf, trace_fr, ml_deltas_dict):
      
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
            plt.axvline(ml_deltas_dict[varnames[i]], color='r', label='ML')
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

def plot_vi_mf_fr_ml_joint(X, y , X_star, f_star, pp_mean_ml, pp_std_ml_nf, pp_mean_mf, pp_mean_fr, lower_mf, upper_mf, lower_fr, upper_fr, title_ml, title_mf, title_fr):
             
             plt.figure()
             plt.plot(X, y, 'ko')
             plt.plot(X_star, f_star, 'k')
             plt.plot(X_star, pp_mean_ml, color='r', label='ML')
             plt.plot(X_star, pp_mean_mf, color='coral',label='MF')
             plt.plot(X_star, pp_mean_fr, color='g', label='FR')
             plt.fill_between(X_star.ravel(), pp_mean_ml - 1.96*pp_std_ml_nf, pp_mean_ml + 1.96*pp_std_ml_nf, color='r', alpha=0.3)
             plt.fill_between(X_star.ravel(), lower_mf, upper_mf, color='coral', alpha=0.3)
             plt.fill_between(X_star.ravel(), lower_fr, upper_fr, color='g', alpha=0.3)
             plt.legend(fontsize='small')
             plt.title('ML vs. MF vs. FR ' + '\n' + 'N = ' + str(len(X)) + '\n' + 'ML  ' + title_ml + '\n' + title_mf + '\n' + title_fr, fontsize='small')
             

def plot_mcmc_deter_joint(X, y, X_star, f_star, post_samples, pp_mean, lower, upper, pred_mean, pred_var, color):
             
              plt.figure()
              plt.plot(X, y, 'ko')
              plt.plot(X_star, f_star, 'k')
              plt.fill_between(X_star.ravel(), lower, upper, color=color, alpha=0.3)
              plt.fill_between(X_star.ravel(), (pred_mean - 1.96*np.sqrt(pred_var)).ravel(), (pred_mean + 1.96*np.sqrt(pred_var)).ravel(), alpha=0.3, color='b')
              plt.plot(X_star, post_samples.T, 'grey', alpha=0.4)
              plt.plot(X_star, pred_mean, color='b', label='VI DET')
              plt.plot(X_star, pp_mean, color=color, label='VI MCMC')
              plt.legend(fontsize='x-small')
              plt.title('VI MCMC vs. VI DET')


        

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
    
      X_5, y_5, X_star_5, f_star_5 = load_datasets(data_path, 5)
      X_10, y_10, X_star_10, f_star_10 = load_datasets(data_path, 10)
      X_20, y_20, X_star_20, f_star_20 = load_datasets(data_path, 20)
      X_40, y_40, X_star_40, f_star_40 = load_datasets(data_path, 40)
      

     rev_name_mapping = {'ls' : 'log_ls_interval__', 
        'sig_sd' : 'log_s_interval__', 
         'noise_sd': 'log_n_interval__'}
     
     raw_mapping = {'log_ls_interval__':  generative_model(X=X_20, y=y_20).log_ls_interval__, 
              'log_s_interval__': generative_model(X=X_20, y=y_20).log_s_interval__, 
              'log_n_interval__':  generative_model(X=X_20, y=y_20).log_n_interval__}
     
     name_mapping = {'log_ls_interval__':  'ls', 
              'log_s_interval__': 'sig_sd', 
              'log_n_interval__':  'noise_sd'}
          
      # ML 
      
      gpr_5, pp_mean_ml_5, pp_std_ml_5, pp_std_ml_nf_5, rmse_ml_5, lpd_ml_5, ml_deltas_dict_5, title_5 =  get_ml_report(X_5, y_5, X_star_5, f_star_5)
      gpr_10, pp_mean_ml_10, pp_std_ml_10, pp_std_ml_nf_10, rmse_ml_10, lpd_ml_10, ml_deltas_dict_10, title_10 =  get_ml_report(X_10, y_10, X_star_10, f_star_10)
      gpr_20, pp_mean_ml_20, pp_std_ml_20, pp_std_ml_nf_20, rmse_ml_20, lpd_ml_20, ml_deltas_dict_20, title_20 =  get_ml_report(X_20, y_20, X_star_20, f_star_20)
      gpr_40, pp_mean_ml_40, pp_std_ml_40, pp_std_ml_nf_40, rmse_ml_40, lpd_ml_40, ml_deltas_dict_40, title_40 =  get_ml_report(X_40, y_40, X_star_40, f_star_40)     
      
      # VI - Mean Field and Full-Rank
      
      def variational_fitting(X, y, iterations):
      
            with generative_model(X=X, y=y):
                   
                  mf = pm.ADVI()
                  fr = pm.FullRankADVI()
            
                  tracker_mf = pm.callbacks.Tracker(
                  mean = mf.approx.mean.eval,    
                  std = mf.approx.std.eval)
                  
                  tracker_fr = pm.callbacks.Tracker(
                  mean = fr.approx.mean.eval,    
                  std = fr.approx.std.eval)
            
                  mf.fit(callbacks=[tracker_mf], n=iterations)
                  fr.fit(callbacks=[tracker_fr], n=iterations)
                  
                  trace_mf = mf.approx.sample(5000)
                  trace_fr = fr.approx.sample(5000)
            
            return  mf, fr, trace_mf, trace_fr, tracker_mf, tracker_fr
      
      def variational_post_processing(mf, fr, trace_mf, trace_fr, name_mapping):
            
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
            
            update_param_dict(mf, mf_param, pm.summary(trace_mf), name_mapping)
            update_param_dict(fr, fr_param, pm.summary(trace_fr), name_mapping)
            
            return trace_mf_df, trace_fr_df, mf_param, fr_param, means_mf, std_mf, means_fr, std_fr
      
      # Variational fitting

      mf_5, fr_5, trace_mf_5, trace_fr_5, tracker_mf_5, tracker_fr_5 = variational_fitting(X_5, y_5, 60000)
      mf_10, fr_10, trace_mf_10, trace_fr_10, tracker_mf_10, tracker_fr_10 = variational_fitting(X_10, y_10, 80000)
      mf_20, fr_20, trace_mf_20, trace_fr_20, tracker_mf_20, tracker_fr_20 = variational_fitting(X_20, y_20, 45000)
      mf_40, fr_40, trace_mf_40, trace_fr_40, tracker_mf_40, tracker_fr_40 = variational_fitting(X_40, y_40, 25000)
      
      # Variational post-processing
      
      trace_mf_df_5, trace_fr_df_5, mf_param_5, fr_param_5, means_mf_5, std_mf_5, means_fr_5, std_fr_5 = variational_post_processing(mf_5, fr_5, trace_mf_5, trace_fr_5, name_mapping)
      trace_mf_df_10, trace_fr_df_10, mf_param_10, fr_param_10, means_mf_10, std_mf_10, means_fr_10, std_fr_10 = variational_post_processing(mf_10, fr_10, trace_mf_10, trace_fr_10, name_mapping)
      trace_mf_df_20, trace_fr_df_20, mf_param_20, fr_param_20, means_mf_20, std_mf_20, means_fr_20, std_fr_20 = variational_post_processing(mf_20, fr_20, trace_mf_20, trace_fr_20, name_mapping)
      trace_mf_df_40, trace_fr_df_40, mf_param_40, fr_param_40, means_mf_40, std_mf_40, means_fr_40, std_fr_40 = variational_post_processing(mf_40, fr_40, trace_mf_40, trace_fr_40, name_mapping)

      trace_mf_df_20.to_csv(results_path + 'trace_mf_df_n20.csv', sep=',')
      trace_fr_df_20.to_csv(results_path + 'trace_fr_df_n20.csv', sep=',')
      
      trace_mf_df_40.to_csv(results_path + 'trace_mf_df_n40.csv', sep=',')
      trace_fr_df_40.to_csv(results_path + 'trace_fr_df_n40.csv', sep=',')
      
      # Variational parameters track  
      
      convergence_report(tracker_mf_5, tracker_fr_5, mf_param_5, fr_param_5, true_hyp, varnames, rev_name_mapping, raw_mapping, name_mapping)
      convergence_report(tracker_mf_10, tracker_fr_10, mf_param_10, fr_param_10, true_hyp, varnames, rev_name_mapping,  raw_mapping, name_mapping)
      convergence_report(tracker_mf_20, tracker_fr_20, mf_param_20, fr_param_20, true_hyp, varnames, rev_name_mapping,  raw_mapping, name_mapping)
      convergence_report(tracker_mf_40, tracker_fr_40, mf_param_40, fr_param_40, true_hyp, varnames, rev_name_mapping,  raw_mapping, name_mapping)

      # ELBO Convergence
      
      plot_elbo_convergence(mf_5, fr_5)
      plot_elbo_convergence(mf_10, fr_10)
      plot_elbo_convergence(mf_20, fr_20)
      plot_elbo_convergence(mf_40, fr_40)

      # Traceplot with Mean-Field and Full-Rank
      
      trace_report(mf_5, fr_5, means_mf_5, std_mf_5, means_fr_5, std_fr_5, true_hyp, trace_mf_5, trace_fr_5, ml_deltas_dict_5)
      trace_report(mf_10, fr_10, means_mf_10, std_mf_10, means_fr_10, std_fr_10, true_hyp, trace_mf_10, trace_fr_10, ml_deltas_dict_10)
      trace_report(mf_20, fr_20, means_mf_20, std_mf_20, means_fr_20, std_fr_20, true_hyp, trace_mf_20, trace_fr_20, ml_deltas_dict_20)
      trace_report(mf_40, fr_40, means_mf_40, std_mf_40, means_fr_40, std_fr_40, true_hyp, trace_mf_40, trace_fr_40, ml_deltas_dict_40)

      # Bi-variate relationship
      
      bi_list = []
      for i in combinations(varnames, 2):
          bi_list.append(i)

      
      get_bivariate_hyp(bi_list, trace_mf_5, trace_fr_5)
      get_bivariate_hyp(bi_list, trace_mf_10, trace_fr_10)
      get_bivariate_hyp(bi_list, trace_mf_20, trace_fr_20)
      get_bivariate_hyp(bi_list, trace_mf_40, trace_fr_40)

      
      #------------------------------Predictions with VI MCMC---------------------------------------------
      
      # N = 40
      
      thin_factor=100
      write_posterior_predictive_samples(trace_mf_40, thin_factor, X_40, y_40, X_star_40, results_path, 'mf')
      write_posterior_predictive_samples(trace_fr_40, thin_factor, X_40, y_40, X_star_40, results_path, 'fr')

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

      title_mf_40 = 'MF  ' +  'RMSE: ' + str(rmse_mf_40) + ' ' + 'NLPD: ' + str(-lpd_mf_40)
      title_fr_40 = 'FR  ' +  'RMSE: ' + str(rmse_fr_40) + ' ' + 'NLPD: ' + str(-lpd_fr_40)
      
       # N = 20
      
      thin_factor=100
      write_posterior_predictive_samples(trace_mf_20, thin_factor, X_20, y_20, X_star_20, results_path, 'mf')
      write_posterior_predictive_samples(trace_fr_20, thin_factor, X_20, y_20, X_star_20, results_path, 'fr')

      post_means_mf_20, post_stds_mf_20 = load_post_samples(results_path + 'means_mf_20.csv', results_path + 'std_mf_20.csv')
      post_means_fr_20, post_stds_fr_20 = load_post_samples(results_path + 'means_fr_20.csv', results_path + 'std_fr_20.csv')
       
      u20 = np.ones(len(post_means_mf_20))

      pp_mean_mf_20 = get_posterior_predictive_mean(post_means_mf_20, weights=None)
      pp_mean_fr_20 = get_posterior_predictive_mean(post_means_fr_20, weights=None)
       
      lower_mf_20, upper_mf_20 = get_posterior_predictive_uncertainty_intervals(post_means_mf_20, post_stds_mf_20, u20)
      lower_fr_20, upper_fr_20 = get_posterior_predictive_uncertainty_intervals(post_means_fr_20, post_stds_fr_20, u20)
       
      rmse_mf_20 = rmse(pp_mean_mf_20, f_star_20)
      rmse_fr_20 = rmse(pp_mean_fr_20, f_star_20)

      lppd_mf_20, lpd_mf_20 = log_predictive_mixture_density(f_star_20, post_means_mf_20, post_stds_mf_20, None)
      lppd_fr_20, lpd_fr_20 = log_predictive_mixture_density(f_star_20, post_means_fr_20, post_stds_fr_20, None)

      title_mf_20 = 'MF   ' +  'RMSE: ' + str(rmse_mf_20) + ' ' + 'NLPD: ' + str(-lpd_mf_20)
      title_fr_20 =  'FR  ' + 'RMSE: ' + str(rmse_fr_20) + ' ' + 'NLPD: ' + str(-lpd_fr_20)
       
       # N = 10
      
      thin_factor=100
      write_posterior_predictive_samples(trace_mf_10, thin_factor, X_10, y_10, X_star_10, results_path, 'mf')
      write_posterior_predictive_samples(trace_fr_10, thin_factor, X_10, y_10, X_star_10, results_path, 'fr')

      post_means_mf_10, post_stds_mf_10 = load_post_samples(results_path + 'means_mf_10.csv', results_path + 'std_mf_10.csv')
      post_means_fr_10, post_stds_fr_10 = load_post_samples(results_path + 'means_fr_10.csv', results_path + 'std_fr_10.csv')
       
      u10 = np.ones(len(post_means_mf_10))

      pp_mean_mf_10 = get_posterior_predictive_mean(post_means_mf_10, weights=None)
      pp_mean_fr_10 = get_posterior_predictive_mean(post_means_fr_10, weights=None)
       
      lower_mf_10, upper_mf_10 = get_posterior_predictive_uncertainty_intervals(post_means_mf_10, post_stds_mf_10, u10)
      lower_fr_10, upper_fr_10 = get_posterior_predictive_uncertainty_intervals(post_means_fr_10, post_stds_fr_10, u10)
       
      rmse_mf_10 = rmse(pp_mean_mf_10, f_star_10)
      rmse_fr_10 = rmse(pp_mean_fr_10, f_star_10)

      lppd_mf_10, lpd_mf_10 = log_predictive_mixture_density(f_star_10, post_means_mf_10, post_stds_mf_10, None)
      lppd_fr_10, lpd_fr_10 = log_predictive_mixture_density(f_star_10, post_means_fr_10, post_stds_fr_10, None)

      title_mf_10 = 'MF  ' + 'RMSE: ' + str(rmse_mf_10) + '  ' + 'NLPD: ' + str(-lpd_mf_10)
      title_fr_10 = 'FR  ' + 'RMSE: ' + str(rmse_fr_10) + '  ' + 'NLPD: ' + str(-lpd_fr_10)
      
      
      # N = 5
      
      thin_factor=100
      write_posterior_predictive_samples(trace_mf_5, thin_factor, X_5, y_5, X_star_5, results_path, 'mf')
      write_posterior_predictive_samples(trace_fr_5, thin_factor, X_5, y_5, X_star_5, results_path, 'fr')

      post_means_mf_5, post_stds_mf_5 = load_post_samples(results_path + 'means_mf_5.csv', results_path + 'std_mf_5.csv')
      post_means_fr_5, post_stds_fr_5 = load_post_samples(results_path + 'means_fr_5.csv', results_path + 'std_fr_5.csv')
       
      u5 = np.ones(len(post_means_mf_5))

      pp_mean_mf_5 = get_posterior_predictive_mean(post_means_mf_5, weights=None)
      pp_mean_fr_5 = get_posterior_predictive_mean(post_means_fr_5, weights=None)
       
      lower_mf_5, upper_mf_5 = get_posterior_predictive_uncertainty_intervals(post_means_mf_5, post_stds_mf_5, u5)
      lower_fr_5, upper_fr_5 = get_posterior_predictive_uncertainty_intervals(post_means_fr_5, post_stds_fr_5, u5)
       
      rmse_mf_5 = rmse(pp_mean_mf_10, f_star_10)
      rmse_fr_5 = rmse(pp_mean_fr_10, f_star_10)

      lppd_mf_5, lpd_mf_5 = log_predictive_mixture_density(f_star_5, post_means_mf_5, post_stds_mf_5, None)
      lppd_fr_5, lpd_fr_5 = log_predictive_mixture_density(f_star_5, post_means_fr_5, post_stds_fr_5, None)

      title_mf_5 = 'MF  ' + 'RMSE: ' + str(rmse_mf_5) + '  ' + 'NLPD: ' + str(-lpd_mf_5)
      title_fr_5 = 'FR  ' + 'RMSE: ' + str(rmse_fr_5) + '  ' + 'NLPD: ' + str(-lpd_fr_5)
      

      plot_vi_mf_fr_ml_joint(X_5, y_5, X_star_5, f_star_5, pp_mean_ml_5, pp_std_ml_nf_5, pp_mean_mf_5, pp_mean_fr_5, lower_mf_5, upper_mf_5, lower_fr_5, upper_fr_5, title_5[70:], title_mf_5, title_fr_5)  
      plot_vi_mf_fr_ml_joint(X_10, y_10, X_star_10, f_star_10, pp_mean_ml_10, pp_std_ml_nf_10, pp_mean_mf_10, pp_mean_fr_10, lower_mf_10, upper_mf_10, lower_fr_10, upper_fr_10, title_10[70:], title_mf_10, title_fr_10)  
      plot_vi_mf_fr_ml_joint(X_20, y_20, X_star_20, f_star_20, pp_mean_ml_20, pp_std_ml_nf_20, pp_mean_mf_20, pp_mean_fr_20, lower_mf_20, upper_mf_20, lower_fr_20, upper_fr_20, title_20[70:], title_mf_20, title_fr_20)
      plot_vi_mf_fr_ml_joint(X_40, y_40, X_star_40, f_star_40, pp_mean_ml_40, pp_std_ml_nf_40, pp_mean_mf_40, pp_mean_fr_40, lower_mf_40, upper_mf_40, lower_fr_40, upper_fr_40, title_40[67:], title_mf_40, title_fr_40)
      
      
      #---------------------Full deterministic prediction------------------------------------------------------
      
      # Extracting mu_theta and cov_theta
      
      cov_theta_mf_5 =  np.cov(trace_mf_df_5[varnames], rowvar=False)
      cov_theta_fr_5 = np.cov(trace_fr_df_5[varnames], rowvar=False)
      
      mu_theta_mf_5 = mf_param_5['mu_implicit']
      mu_theta_fr_5 = fr_param_5['mu_implicit']
      
      cov_theta_mf_10 =  np.cov(trace_mf_df_10[varnames], rowvar=False)
      cov_theta_fr_10 = np.cov(trace_fr_df_10[varnames], rowvar=False)
      
      mu_theta_mf_10 = mf_param_10['mu_implicit']
      mu_theta_fr_10 = fr_param_10['mu_implicit']
      
      cov_theta_mf_20 =  np.cov(trace_mf_df_20[varnames], rowvar=False)
      cov_theta_fr_20 = np.cov(trace_fr_df_20[varnames], rowvar=False)
      
      mu_theta_mf_20 = mf_param_20['mu_implicit']
      mu_theta_fr_20 = fr_param_20['mu_implicit']
      
      cov_theta_mf_40 =  np.cov(trace_mf_df_40[varnames], rowvar=False)
      cov_theta_fr_40 = np.cov(trace_fr_df_40[varnames], rowvar=False)
      
      mu_theta_mf_40 = mf_param_40['mu_implicit']
      mu_theta_fr_40 = fr_param_40['mu_implicit']
      
      # Persist mu and cov from VI 
      
      np.savetxt(results_path + 'mu_theta_fr_20.csv', pd.Series(mu_theta_fr_20), delimiter=',')
      np.savetxt(results_path + 'cov_theta_fr_20.csv', cov_theta_fr_20, delimiter=',')
      
      np.savetxt(results_path + 'mu_theta_mf_20.csv', pd.Series(mu_theta_mf_20), delimiter=',')
      np.savetxt(results_path + 'cov_theta_mf_20.csv', cov_theta_mf_20, delimiter=',')
      
      np.savetxt(results_path + 'mu_theta_fr_40.csv', pd.Series(mu_theta_fr_20), delimiter=',')
      np.savetxt(results_path + 'cov_theta_fr_40.csv', cov_theta_fr_20, delimiter=',')
      
      np.savetxt(results_path + 'mu_theta_mf_40.csv', pd.Series(mu_theta_mf_40), delimiter=',')
      np.savetxt(results_path + 'cov_theta_mf_40.csv', cov_theta_mf_40, delimiter=',')
      
      # Loading VI variational parameters
      
      mu_theta_fr_20 = pd.read_csv(results_path + 'mu_theta_fr_40.csv', header=None)
      mu_theta_fr_20 = dict(zip(list(varnames), list(mu_theta_fr_20[0])))
      cov_theta_fr_20 = np.array(pd.read_csv(results_path + 'cov_theta_fr_20.csv', header=None))
      
      mu_theta = mu_theta_fr_40
      cov_theta = cov_theta_fr_40
      
      # Automatic differentiation
      
      theta = np.array([mu_theta['sig_sd'], mu_theta['ls'], mu_theta['noise_sd']])

      def kernel(theta, X1, X2):
            ''' Isotropic squared exponential kernel. 
            Computes a covariance matrix from points in X1 and X2. 
            Args: X1: Array    of m points (m x d). X2: Array of n points (n x d). 
            Returns: Covariance matrix (m x n). '''
            
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
            return theta[0]**2 * np.exp(-0.5 / theta[1]**2 * sqdist)
        
      def gp_mean(theta, X, y, x_star):
              
              K = kernel(theta, X, X)
              K_noise = K + theta[2]**2 * np.eye(len(X))
              K_s = kernel(theta, X, x_star)
              return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)
        
      dh = elementwise_grad(gp_mean)
      d2h = jacobian(dh)
        #d2hh = hessian(gp_mean)
        
      def get_vi_analytical_mean(X, y, X_star, d2h, theta, mu_theta, cov_theta):
                  
          K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), mu_theta)            
          pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
            #pred_vi_var =  K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)

          pred_mean = []
          pred_var = []
      
          for i in np.arange(len(X_star)):
                
                x_star = X_star[i].reshape(1,1)
      
                pred_mean.append(pred_vi_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), cov_theta)))

          return pred_mean
      
      pred_mean_ng_40 = get_vi_analytical_mean(X_40, y_40, X_star_40, d2h, theta, mu_theta, cov_theta)
      
    
      
    
        # MF - MCVI vs DVI
     
       plot_mcmc_deter_joint(X_5, y_5, X_star_5, f_star_5, post_means_mf_5, pp_mean_mf_5, lower_mf_5, upper_mf_5, pred_mean_mf_5, pred_var_mf_5, 'coral')
       plot_mcmc_deter_joint(X_10, y_10, X_star_10, f_star_10, post_means_mf_10, pp_mean_mf_10, lower_mf_10, upper_mf_10, pred_mean_mf_10, pred_var_mf_10, 'coral')
       plot_mcmc_deter_joint(X_20, y_20, X_star_20, f_star_20, post_means_mf_20, pp_mean_mf_20, lower_mf_20, upper_mf_20, pred_mean_mf_20, pred_var_mf_20, 'coral')
        plot_mcmc_deter_joint(X_40, y_40, X_star_40, f_star_40, post_means_mf_40, pp_mean_mf_40, lower_mf_40, upper_mf_40, pred_mean_mf_40, pred_var_mf_40, 'coral')
      
        # FR - MCVI vs DVI
      
      plot_mcmc_deter_joint(X_5, y_5, X_star_5, f_star_5, post_means_fr_5, pp_mean_fr_5, lower_fr_5, upper_fr_5, pred_mean_fr_5, pred_var_fr_5, 'g')
      plot_mcmc_deter_joint(X_10, y_10, X_star_10, f_star_10, post_means_fr_10, pp_mean_fr_10, lower_fr_10, upper_fr_10, pred_mean_fr_10, pred_var_fr_10, 'g')
     plot_mcmc_deter_joint(X_20, y_20, X_star_20, f_star_20, post_means_fr_20, pp_mean_fr_20, lower_fr_20, upper_fr_20, pred_mean_fr_20, pred_var_fr_20, 'g')
      plot_mcmc_deter_joint(X_40, y_40, X_star_40, f_star_40, post_means_fr_40, pp_mean_fr_40, lower_fr_40, upper_fr_40, pred_mean_fr_40, pred_var_fr_40, 'g')
      

