r#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:29:02 2019

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
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic as RQ, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa


def get_subset_trace(trace, varnames):
      
      trace_df = pd.DataFrame()
      for i in varnames:
            trace_df[i] = trace.get_values(i)
      return trace_df

# Implicit variational posterior density
            
def get_implicit_variational_posterior(var, means, std, x):
      
      sigmoid = lambda x : 1 / (1 + np.exp(-x))
      
      if (var.name[-2:] == '__'):
            # Then it is an interval variable
            
            eps = lambda x : var.distribution.transform_used.forward_val(np.log(x))
            backward_theta = lambda x: var.distribution.transform_used.backward(x).eval()   
            width = (var.distribution.transform_used.b -  var.distribution.transform_used.a).eval()
            total_jacobian = lambda x: x*(width)*sigmoid(eps(x))*(1-sigmoid(eps(x)))
            pdf = lambda x: st.norm.pdf(eps(x), means[var.name], std[var.name])/total_jacobian(x)
            return pdf(x)
      
      else:
            # Then it is just a log variable
            
            pdf = lambda x: st.norm.pdf(np.log(x), means[var.name], std[var.name])/x   
            return pdf(x)
            

# Converting raw params back to param space

def analytical_variational_opt(model, param_dict, summary_trace):
      
      keys = list(param_dict['mu'].keys())
      
      # First tackling transformed means
      
      mu_implicit = {}
      rho_implicit = {}
      for i in keys:
            if (i[-2:] == '__'):
                  name = name_mapping[i]
                  mean_value = np.exp(raw_mapping.get(i).distribution.transform_used.backward(param_dict['mu'][i]).eval())
                  sd_value = summary_trace['sd'][name]
                  mu_implicit.update({name : np.array(mean_value)})
                  rho_implicit.update({name : np.array(sd_value)})
            else:
                  name = name_mapping[i]
                  mean_value = np.exp(param_dict['mu'][i])
                  sd_value = summary_trace['sd'][name]
                  name = name_mapping[i]
                  mu_implicit.update({name : np.array(mean_value)})
                  rho_implicit.update({name : np.array(sd_value)})
      param_dict.update({'mu_implicit' : mu_implicit})
      param_dict.update({'rho_implicit' : rho_implicit})

      return param_dict

def get_empirical_covariance(trace, varnames):
      
      df = pm.trace_to_dataframe(trace)
      return pd.DataFrame(np.cov(df[varnames], rowvar=False), index=varnames, columns=varnames)


def transform_tracker_values(tracker, param_dict):

      mean_df = pd.DataFrame(np.array(tracker['mean']), columns=list(param_dict['mu'].keys()))
      sd_df = pd.DataFrame(np.array(tracker['std']), columns=list(param_dict['mu'].keys()))
      for i in mean_df.columns:
            print(i)
            if (i[-2:] == '__'):
                 mean_df[name_mapping[i]] = np.exp(raw_mapping.get(i).distribution.transform_used.backward(mean_df[i]).eval()) 
            else:
                mean_df[name_mapping[i]] = np.exp(mean_df[i]) 
      return mean_df, sd_df
                
def convergence_report(tracker, param_dict, elbo, title):
      
      # Plot Negative ElBO track with params in true space
      
      mean_mf_df, sd_mf_df = transform_tracker_values(tracker_mf, mf_param)
      mean_fr_df, sd_fr_df = transform_tracker_values(tracker_fr, fr_param)

      fig = plt.figure(figsize=(16, 9))
      for i in np.arange(3):
            print(i)
#            if (np.mod(i,8) == 0):
#                   fig = plt.figure(figsize=(16,9))
#                   i = i + 8
#                   print(i)
            plt.subplot(2,4,np.mod(i, 8)+1)
            plt.plot(mean_mf_df[varnames[i+8]], color='coral')
            plt.plot(mean_fr_df[varnames[i+8]], color='green')
            plt.title(varnames[i+8])
            plt.axhline(param_dict['mu_implicit'][varnames[i+8]], color='r')
      
      
      fig = plt.figure(figsize=(16, 9))
      mu_ax = fig.add_subplot(221)
      std_ax = fig.add_subplot(222)
      hist_ax = fig.add_subplot(212)
      mu_ax.plot(tracker['mean'])
      mu_ax.set_title('Mean track')
      std_ax.plot(tracker['std'])
      std_ax.set_title('Std track')
      hist_ax.plot(elbo)
      hist_ax.set_title('Negative ELBO track');
      fig.suptitle(title)

# Constructing posterior predictive distribution

def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):
      
      # Fixed at 95% CI
      
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True)
            mixture_draws = np.array([st.norm.rvs(loc=sample_means.iloc[j,i], scale=sample_stds.iloc[j,i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5,97.5])
            lower_.append(lower)
            upper_.append(upper)
      return np.array(lower_), np.array(upper_)

def get_posterior_predictive_mean(sample_means):
      
      return np.mean(sample_means)

def get_posterior_predictive_samples(trace, thin_factor, X_star, path, method):
      
      means_file = path + 'means_' + method + '.csv'
      std_file = path + 'std_' + method + '.csv'
    
      means_writer = csv.writer(open(means_file, 'w')) 
      std_writer = csv.writer(open(std_file, 'w'))
      
      means_writer.writerow(df['year'][sep_idx:])
      std_writer.writerow(df['year'][sep_idx:])
      
      for i in np.arange(len(trace))[::thin_factor]:
            
            print('Predicting ' + str(i))
            mu, var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            std = np.sqrt(var)
            
            print('Writing out ' + str(i) + ' predictions')
            means_writer.writerow(np.round(mu, 3))
            std_writer.writerow(np.round(std, 3))

      time.sleep(5.0)    # pause 5.5 seconds
      sample_means =  pd.read_csv(means_file, sep=',', header=0)
      sample_stds = pd.read_csv(std_file, sep=',', header=0)
      
      return sample_means, sample_stds

if __name__ == "__main__":


      varnames = ['s_1', 'ls_2', 's_3', 'ls_4', 'ls_5', 's_6', 'ls_7', 'alpha_8', 's_9', 'ls_10', 'n_11']
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Co2/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Co2/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Co2/'

      
      path = uni_path
      
      df = pd.read_table(path + 'mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
      
      # creat a date index for the data - convert properly from the decimal year 
      
      #df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')
      
      df.dropna(inplace=True)
      
      mean_co2 = df['co2'][0]
      std_co2 = np.std(df['co2'])   
      
      # normalize co2 levels
         
      #y = normalize(df['co2'])
      y = df['co2']
      t = df['year'] - df['year'][0]
      
      sep_idx = 545
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      
      # sklearn kernel 
      
      # se +  sexper + rq + se + noise
      
      sk1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
      sk2 = 2.0**2 * RBF(length_scale=100.0) \
          * PER(length_scale=1.0, periodicity=1.0,
                           periodicity_bounds="fixed")  # seasonal component
      # medium term irregularities
      sk3 = 0.5**2 * RQ(length_scale=1.0, alpha=1.0)
      sk4 = 0.1**2 * RBF(length_scale=0.1) \
          + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, np.inf))  # noise terms
      #---------------------------------------------------------------------
          
      # Type II ML for hyp.
          
      #---------------------------------------------------------------------
          
      sk_kernel = sk1 + sk2 + sk3 + sk4
      gpr = GaussianProcessRegressor(kernel=sk_kernel, normalize_y=True)
      
      # Fit to data 
      
      gpr.fit(t_train, y_train)
           
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f"
      % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training data")
      
      mu_fit, std_fit = gpr.predict(t_train, return_std=True)
      
      print("Predicting with trained gp on test data")
      
      mu_test, std_test = gpr.predict(t_test, return_std=True)
      mu_test, cov_test = gpr.predict(t_test, return_cov=True)
      
      rmse_ = pa.rmse(mu_test, y_test)
      
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      
      plt.figure()
      plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
      plt.plot(df['year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
      plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
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
      
      ml_values = [ml_deltas[v] for v in varnames]
      
      ml_df = pd.DataFrame(data=np.column_stack((varnames, ml_values)), columns=['hyp','values'])
      
      ml_df.to_csv(results_path + 'co2_ml.csv', sep=',')
      
      # Reading pre-stored values
      
      ml_df = pd.read_csv(results_path + 'co2_ml.csv')
      
      ml_deltas = dict(zip(ml_df['hyp'], ml_df['values']))
      
      #---------------------------------------------------------------------
          
      # Vanilla GP - pymc3
          
      #---------------------------------------------------------------------

      with pm.Model() as model:
        
             # Specify the covariance function
       
             k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2) 
             k2 = pm.gp.cov.Constant(s_3**2)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
             k3 = pm.gp.cov.Constant(s_6**2)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
             k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_7) +  pm.gp.cov.WhiteNoise(n_11**2)
             
             gp_trend = pm.gp.Marginal(mean_func=pm.gp.mean.Constant(c=np.mean(y_train)), cov_func=k1)
             gp_periodic = pm.gp.Marginal(cov_func=k2)
             gp_rational = pm.gp.Marginal(cov_func=k3)
             gp_noise = pm.gp.Marginal(cov_func=k4)
      
             k =  k1 + k2 + k3
                
             gp = gp_trend + gp_periodic + gp_rational
            
             # Marginal Likelihood
             y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=k4)
        
      with model:
          
            f_cond = gp.conditional("f_cond", Xnew=t_test)

      post_pred_mean, post_pred_cov = gp.predict(t_test, pred_noise=True)
      post_pred_mean, post_pred_cov_nf = gp.predict(t_test, pred_noise=False)
      post_pred_std = np.sqrt(np.diag(post_pred_cov))

      plt.figure()
      plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
      plt.plot(df['year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      #plt.plot(df['year'][sep_idx:], post_pred_mean, alpha=0.5, color='b')
      plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
      plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')

     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------


with pm.Model() as co2_model:
  
      # prior on lengthscales
       
       log_l2 = pm.Uniform('log_l2', lower=-5, upper=10, testval=np.log(ml_deltas['ls_2']))
       log_l4 = pm.Uniform('log_l4', lower=-5, upper=5, testval=np.log(ml_deltas['ls_4']))
       log_l5 = pm.Uniform('log_l5', lower=-1, upper=1, testval=np.log(ml_deltas['ls_5']))
       log_l7 = pm.Uniform('log_l7', lower=-1, upper=2, testval=np.log(ml_deltas['ls_7']))
       log_l10 = pm.Uniform('log_l10', lower=-10, upper=-1, testval=np.log(ml_deltas['ls_10']))

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
       log_s3 = pm.Uniform('log_s3', lower=-2, upper=3, testval=np.log(ml_deltas['s_3']))
       log_s6 = pm.Normal('log_s6', mu=np.log(ml_deltas['s_6']), sd=1)
       log_s9 = pm.Uniform('log_s9', lower=-10, upper=-1, testval=np.log(ml_deltas['s_9']))

       s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_3 = pm.Deterministic('s_3', tt.exp(log_s3))
       s_6 = pm.Deterministic('s_6', tt.exp(log_s6))
       s_9 = pm.Deterministic('s_9', tt.exp(log_s9))
       
       #s_3 = 2.59
       #s_9 = 0.169
      
       # prior on alpha
      
       log_alpha8 = pm.Normal('log_alpha8', mu=np.log(ml_deltas['alpha_8']), sd=0.05)
       alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
       #alpha_8 = 0.121
       
       # prior on noise variance term
      
       log_n11 = pm.Uniform('log_n11', lower=-2, upper=5, testval=np.log(ml_deltas['n_11']))
       n_11 = pm.Deterministic('n_11', tt.exp(log_n11))
       
       #n_11 = 0.195
       
       # Specify the covariance function
       
       k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2) 
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
      
with co2_model:
      
      fr = pm.FullRankADVI()
        
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=15000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)
      

with co2_model:
      
      check_mf = pm.ADVI()
      check_fr = pm.FullRankADVI()
      
      
bij_mf = mf.approx.groups[0].bij
mf_param = {param.name: bij_mf.rmap(param.eval())
	 for param in mf.approx.params}

bij_fr = fr.approx.groups[0].bij
fr_param = {param.name: bij_fr.rmap(param.eval())
	 for param in fr.approx.params}

check_mf.approx.params[0].set_value(bij_mf.map(mf_param['mu']))
check_mf.approx.params[1].set_value(bij_mf.map(mf_param['rho']))

# Updating with implicit values

mf_param = analytical_variational_opt(co2_model, mf_param, pm.summary(trace_mf))
fr_param = analytical_variational_opt(co2_model, fr_param, pm.summary(trace_fr))

# Saving raw ADVI results

mf_df = pd.DataFrame(mf_param)
fr_df = pd.DataFrame(fr_param)

mf_df.to_csv(results_path  + 'VI/mf_df_raw.csv', sep=',')
fr_df.to_csv(results_path + 'VI/fr_df_raw.csv', sep=',')


rv_mapping = {'s_1':  co2_model.log_s1, 
              'ls_2': co2_model.log_l2_interval__, 
              's_3':  co2_model.log_s3_interval__,
              'ls_4': co2_model.log_l4_interval__,
              'ls_5': co2_model.log_l5_interval__,
              's_6': co2_model.log_s6,
              'ls_7': co2_model.log_l7_interval__,
              'alpha_8': co2_model.log_alpha8,
              's_9': co2_model.log_s9_interval__,
              'ls_10': co2_model.log_l10_interval__,
               'n_11': co2_model.log_n11_interval__
                    }

raw_mapping = {'log_s1':  co2_model.log_s1, 
              'log_l2_interval__': co2_model.log_l2_interval__, 
              'log_s3_interval__':  co2_model.log_s3_interval__,
              'log_l4_interval__': co2_model.log_l4_interval__,
              'log_l5_interval__': co2_model.log_l5_interval__,
              'log_s6': co2_model.log_s6,
              'log_l7_interval__': co2_model.log_l7_interval__,
              'log_alpha8': co2_model.log_alpha8,
              'log_s9_interval__': co2_model.log_s9_interval__,
              'log_l10_interval__': co2_model.log_l10_interval__,
               'log_n11_interval__': co2_model.log_n11_interval__ }


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


# Loading persisted results
   
trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/u_prior2/', model=co2_model)


# Traceplots with deltas

pa.traceplots(trace_hmc, varnames, ml_deltas)
pa.traceplots(trace_mf, varnames, ml_deltas)
pa.traceplots(trace_fr, varnames, ml_deltas)


# Covariance matrix

hmc_cov = np.corrcoef(pm.trace_to_dataframe(trace_hmc)[varnames], rowvar=False)
fr_cov = np.corrcoef(pm.trace_to_dataframe(trace_fr)[varnames], rowvar=False)

fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot(121)
ax1.imshow(hmc_cov)
ax1.set_xticks(np.arange(11))
ax1.set_xticklabels(varnames, rotation=70, minor=False)
ax1.set_yticks(np.arange(11))
ax1.set_yticklabels(varnames, minor=False)
ax1.set_title('HMC')
ax2 = plt.subplot(122)
ax2.imshow(fr_cov)
ax2.set_xticks(np.arange(11))
ax2.set_xticklabels(varnames, rotation=70, minor=False)
ax2.set_yticks(np.arange(11))
ax2.set_yticklabels(varnames, minor=False)
ax2.set_title('Full Rank VI')

      
# Predictions

# HMC

sample_means_hmc, sample_stds_hmc = get_posterior_predictive_samples(trace_hmc, 20, t_test, results_path + 'pred_dist/', method='hmc') 

sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc.csv')
sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc.csv')

mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)


# MF

sample_means_mf,sample_stds_mf = get_posterior_predictive_samples(trace_mf, 200, t_test, results_path, method='mf') 

sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')

mu_mf = get_posterior_predictive_mean(sample_means_mf)
lower_mf, upper_mf = get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)


# FR

sample_means_fr, sample_stds_fr = get_posterior_predictive_samples(trace_fr, 100, t_test, results_path, method='fr') 
mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)

sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr.csv')
sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')

lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,30):
      #plt.fill_between(df['year'][sep_idx:], sample_means_hmc.ix[i] - 2*sample_stds_hmc.ix[i],  sample_means_hmc.ix[i] + 2*sample_stds_hmc.ix[i], alpha=0.3, color='grey')
      plt.plot(df['year'][sep_idx:], sample_means_hmc.ix[i], color='b', alpha=0.3)


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,20):
      plt.fill_between(df['year'][sep_idx:], sample_means_mf.ix[i] - 2*sample_stds_mf.ix[i],  sample_means_mf.ix[i] + 2*sample_stds_mf.ix[i], alpha=0.3, color='grey')
      plt.plot(df['year'][sep_idx:], sample_means_mf.ix[i], color='coral', alpha=0.3)
      
plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,30):
     plt.fill_between(df['year'][sep_idx:], sample_means_fr.ix[i] - 2*sample_stds_fr.ix[i],  sample_means_fr.ix[i] + 2*sample_stds_fr.ix[i], alpha=0.3, color='grey')
     plt.plot(df['year'][sep_idx:], sample_means_fr.ix[i], color='green', alpha=0.3)


# Metrics

rmse_hmc = pa.rmse(mu_hmc, y_test)
rmse_mf = pa.rmse(mu_mf, y_test)
rmse_fr = pa.rmse(mu_fr, y_test)

lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
lppd_mf, lpd_mf = pa.log_predictive_mixture_density(y_test, sample_means_mf, sample_stds_mf)
lppd_fr, lpd_fr = pa.log_predictive_mixture_density(y_test, sample_means_fr, sample_stds_fr)

# Plot with HMC + ADVI + Type II results with RMSE and LPD for co2 data

title = 'Type II ML   ' + ' RMSE: ' + str(rmse_)  + '   LPD: ' + str(lpd_) + '\n' +  ' HMC         '  + '    RMSE: ' + str(rmse_hmc) + '   LPD: ' + str(lpd_hmc) + '\n' + ' MF           ' +  '     RMSE: ' + str(rmse_mf) + '    LPD: ' + str(lpd_mf) +  '\n' + ' FR           '  +  '     RMSE: ' + str(rmse_fr)  + '    LPD: ' + str(lpd_fr) 


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], mu_test, alpha=1, label='Type II ML', color='r')
plt.plot(df['year'][sep_idx:], mu_hmc, alpha=1, label='HMC', color='b')
plt.plot(df['year'][sep_idx:], mu_mf, alpha=1, label='MF', color='coral')
plt.plot(df['year'][sep_idx:], mu_fr, alpha=1, label='FR', color='g')

plt.fill_between(df['year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], lower_fr, upper_fr, color='green', alpha=0.5)
plt.fill_between(df['year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.5)
plt.fill_between(df['year'][sep_idx:], (mu_test - 1.96*std_test), (mu_test + 1.96*std_test), color='red', alpha=0.3)
plt.legend(fontsize='x-small')
plt.title(title, fontsize='x-small')
plt.ylim(370,420)


# Subplot scheme for WIML

plt.figure(figsize=(14,6))

plt.subplot(131)
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='test', color='r')
plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
plt.legend(fontsize='small')
plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')

plt.subplot(132)
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], mu_hmc, alpha=0.5, label='test', color='b')
plt.fill_between(df['year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.2)
plt.legend(fontsize='small')
plt.title('HMC' + '\n' + 'RMSE: ' + str(rmse_hmc) + '\n' + 'LPD: ' + str(lpd_hmc), fontsize='small')


plt.subplot(133)
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], mu_fr, alpha=0.5, label='test', color='g')
plt.fill_between(df['year'][sep_idx:], lower_fr, upper_fr, color='green', alpha=0.2)
plt.legend(fontsize='small')
plt.title('VI' + '\n' + 'RMSE: ' + str(rmse_fr) + '\n' + 'LPD: ' + str(lpd_fr), fontsize='small')

# Write out trace summary & autocorrelation plots

model = 'hmc'
trace = trace_hmc

summary_df = pm.summary(trace)
summary_df['Acc Rate'] = np.mean(trace.get_sampler_stats('mean_tree_accept'))
np.round(summary_df,3).to_csv(results_path + 'trace_summary_co2_' + model + '.csv')
      
# TODO Convert summary_df to box-whisker or violen plot

# Pair plots to look at 2-way relationships in hyperparameters

varnames = ['s_1', 'ls_2','s_3', 'ls_4','ls_5','s_6','ls_7','alpha_8','s_9','ls_10','n_11'] 

k1_names = ['s_1', 'ls_2']
k2_names = ['s_3', 'ls_4', 'ls_5']
k3_names = ['s_6', 'ls_7', 'alpha_8']
k4_names = ['s_9', 'ls_10', 'n_11']

trace_k1 = get_subset_trace(trace, k1_names)
trace_k2 = get_subset_trace(trace, k2_names)
trace_k3 = get_subset_trace(trace, k3_names)
trace_k4 = get_subset_trace(trace, k4_names)

#  PairGrid plot  - bivariate relationships
      
def pair_grid_plot(trace_df, ml_deltas, varnames, color):

      g = sns.PairGrid(trace_df, vars=varnames, diag_sharey=False)
      g = g.map_lower(plot_bi_kde, ml_deltas=ml_deltas, color=color)
      g = g.map_diag(plot_hist, color=color)
      g = g.map_upper(plot_scatter, ml_deltas=ml_deltas, color=color)
      
      for i in np.arange(len(varnames)):
            g.axes[i,i].axvline(ml_deltas[g.x_vars[i]], color='r')
    
def plot_bi_kde(x,y, ml_deltas, color, label):
      
      sns.kdeplot(x, y, n_levels=20, color=color, shade=True, shade_lowest=False, bw='silverman')
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      
def plot_hist(x, color, label):
      
      sns.distplot(x, bins=100, color=color, kde=True)

def plot_scatter(x, y, ml_deltas, color, label):
      
      plt.scatter(x, y, c=color, s=0.5, alpha=0.7)
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      
      
pair_grid_plot(trace_k1, ml_deltas, k1_names, color='coral')
pair_grid_plot(trace_k2, ml_deltas, k2_names, color='coral')
pair_grid_plot(trace_k3, ml_deltas, k3_names, color='coral')
pair_grid_plot(trace_k4, ml_deltas, k4_names, color='coral')

# Pair grid catalog

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
      g = sns.kdeplot(trace[i[0]], trace[i[1]], color="g", shade=True, alpha=0.7, bw='silverman')
      g.scatter(trace[i[0]][::20], trace[i[1]][::20], s=0.5, color='g', alpha=0.4)
      g.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
      plt.xlabel(i[0])
      plt.ylabel(i[1])
      plt.tight_layout()
      

for i, j  in zip(bi_list, np.arange(len(bi_list))):
        print(i)
        print(j)
        if np.mod(j,8) == 0:
            fig = plt.figure(figsize=(15,8))
        plt.subplot(2,4,np.mod(j, 8)+1)
        sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=0.9)
        sns.kdeplot(trace_hmc[i[0]], trace_hmc[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
      
       
        
        