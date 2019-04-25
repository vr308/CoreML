#!/usr/bin/env python3
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
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Span, Label, Legend
from bokeh.io import output_notebook
from bokeh.palettes import brewer
import  scipy.stats as st 
import seaborn as sns
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv

def normalize(y):
            
      return (y - np.mean(y))/np.std(y)

def traceplots(trace, varnames, deltas):

      traces_part1 = pm.traceplot(trace, varnames[0:5], lines=deltas)
      traces_part2 = pm.traceplot(trace, varnames[5:], lines=deltas)
      
      for i in np.arange(5):
            
            delta = deltas.get(str(varnames[i]))
            xmax = max(max(trace[varnames[i]]), delta)
            traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part1[i][0].hist(trace[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
            #traces_part1[i][0].axes.set_xlim(xmin, xmax)
            traces_part1[i][0].legend(fontsize='x-small')
      
      for i in np.arange(6):
            
            delta = deltas.get(str(varnames[i+5]))
            traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part2[i][0].hist(trace[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
            #traces_part2[i][0].axes.set_xscale('log')
            traces_part2[i][0].legend(fontsize='x-small')
            
def traceplot_compare(trace_hmc, trace_mf, trace_fr, varnames, deltas):

      traces_part1 = pm.traceplot(trace_hmc, varnames[0:5], lines=deltas)
      traces_part2 = pm.traceplot(trace_hmc, varnames[5:], lines=deltas)
      
      means_mf = mf.approx.bij.rmap(mf.approx.mean.eval())  
      std_mf = mf.approx.bij.rmap(mf.approx.std.eval())  
      
      for i in np.arange(5):
            
            delta = deltas.get(str(varnames[i]))
            xmax = max(max(trace[varnames[i]]), delta)
            xmin = min(min(trace[varnames[i]]), delta)
            traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part1[i][0].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
            traces_part1[i][0].plot(np.linspace(xmin, xmax,1000), get_implicit_variational_posterior(getattr(co2_model, varnames[i]), means_mf, std_mf, np.linspace(xmin, xmax,1000)), color='coral')
            traces_part1[i][0].hist(np.linspace(xmin, xmax,1000), get_implicit_variational_posterior(getattr(co2_model, varnames[i]), means_mf, std_mf, np.linspace(xmin, xmax,1000)), color='coral')
            #traces_part1[i][0].plot(ranges[i], get_implicit_variational_posterior(fr_rv[i], means_fr, std_fr, ranges[i]), color='g')
            #traces_part1[i][0].axes.set_xlim(xmin, xmax)
            traces_part1[i][0].legend(fontsize='x-small')
      
      for i in np.arange(6):
            
            delta = deltas.get(str(varnames[i+5]))
            traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part2[i][0].hist(trace[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
            traces_par2[i][0].plot(ranges[i], get_implicit_variational_posterior(mf_rv[i], means_mf, std_mf, ranges[i]), color='coral')
            #traces_part2[i][0].plot(ranges[i], get_implicit_variational_posterior(fr_rv[i], means_fr, std_fr, ranges[i]), color='g')
            traces_part2[i][0].legend(fontsize='x-small')
            

def get_subset_trace(trace, varnames):
      
      trace_df = pd.DataFrame()
      for i in varnames:
            trace_df[i] = trace.get_values(i)
      return trace_df

# Implicit variational posterior density
      
def get_implicit_variational_posterior(var, means, std, x):
      
      eps = lambda x : var.distribution.transform_used.forward_val(np.log(x))
      backward_theta = lambda x: var.distribution.transform_used.backward(x).eval()   
      width = (var.distribution.transform_used.b -  var.distribution.transform_used.a).eval()
      total_jacobian = lambda x: x*(width)*sigmoid(eps(x))*(1-sigmoid(eps(x)))
      pdf = lambda x: st.norm.pdf(eps(x), means[var.name], std[var.name])/total_jacobian(x)
      return pdf(x)

# Digging
      


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

# Metrics 
      
def rmse(post_mean, y_test):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - y_test))),3)

def log_predictive_density(y_test, list_means, list_stds):
      
      lppd_per_point = []
      for i in np.arange(len(y_test)):
            print(i)
            lppd_per_point.append(st.norm.pdf(y_test[i], list_means[i], list_stds[i]))
      return np.round(np.mean(np.log(lppd_per_point)),3)
            
def log_predictive_mixture_density(y_test, list_means, list_std):
      
      lppd_per_point = []
      for i in np.arange(len(y_test)):
            print(i)
            components = []
            for j in np.arange(len(list_means)):
                  components.append(st.norm.pdf(y_test[i], list_means.iloc[:,i][j], list_std.iloc[:,i][j]))
            lppd_per_point.append(np.mean(components))
      return lppd_per_point, np.round(np.mean(np.log(lppd_per_point)),3)

                  
if __name__ == "__main__":


      varnames = ['s_1', 'ls_2', 's_3', 'ls_4', 'ls_5', 's_6', 'ls_7', 'alpha_8', 's_9', 'ls_10', 'n_11']
      

      home_path = '~/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/'
      
      path = home_path
      
      df = pd.read_table(home_path + 'mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
      
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
      
      rmse_ = np.round(np.sqrt(np.mean(np.square(mu_test - y_test))), 2)
      
      lpd_ = log_predictive_density(y_test, mu_test, std_test)
      
      plt.figure()
      plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
      #plt.plot(df['year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      #plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
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
      
      ml_df = pd.DataFrame(data=ml_deltas, index=['ml'])
      
      ml_df.to_csv(path + 'co2_ml.csv', sep=',')
      
      ml_df = pd.read_csv(path + 'co2_ml.csv', index_col=None)


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
       k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11**2)

       k =  k1 + k2 + k3 + k4
          
       gp = pm.gp.Marginal(cov_func=k)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=n_11)
              
with co2_model:
      
      # HMC Nuts auto-tuning implementation

      trace_hmc = pm.sample(draws=700, tune=500, chains=1)
            
with co2_model:
    
      pm.save_trace(trace_hmc, directory = path + 'Traces_pickle_hmc/u_prior3/', overwrite=True)
        
with co2_model:
      
      mf = pm.ADVI()

      tracker_mf = pm.callbacks.Tracker(
      mean = mf.approx.mean.eval,    
      std = mf.approx.std.eval)
     
      mf.fit(n=8000, callbacks=[tracker_mf])
      
      trace_mf = mf.approx.sample(4000)
      
with co2_model:
      
      fr = pm.FullRankADVI()
        
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(4000)


bij_mf = mf.approx.groups[0].bij
mf_param = {param.name: bij_mf.rmap(param.eval())
	 for param in mf.approx.params}

bij_fr = fr.approx.groups[0].bij
fr_param = {param.name: bij_fr.rmap(param.eval())
	 for param in fr.approx.params}


with co2_model:
      
   pm.save_trace(trace_mf, directory = path + 'Traces_pickle_advi/u_prior_mf/', overwrite=True)
   pm.save_trace(trace_fr, directory = path + 'Traces_pickle_advi/u_prior_fr/', overwrite=True)


# Loading persisted results
   
trace_hmc_load = pm.load_trace(path + 'Traces_pickle_hmc/u_prior2/', model=co2_model)
trace_mf_load = pm.load_trace(path + 'Traces_pickle_advi/u_prior_mf/', model=co2_model)
trace_advi_load = pm.load_trace(path + 'Traces_pickle_advi/u_prior_fr/', model=co2_model)


# Traceplots with deltas

traceplots(trace_hmc, varnames, ml_deltas)
traceplots(trace_mf, varnames, ml_deltas)
traceplots(trace_fr, varnames, ml_deltas)

      
results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Results/Co2/'

# Get HMC results

sample_means_hmc, sample_stds_hmc = get_posterior_predictive_samples(trace_hmc, 20, t_test, results_path + 'pred_dist/', method='hmc') 

sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc.csv')
sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc.csv')

mu_hmc = get_posterior_predictive_mean(sample_means_hmc)
lower_hmc, upper_hmc = get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)

# Get ADVI results 

# MF

sample_means_mf,sample_stds_mf = get_posterior_predictive_samples(trace_mf, 100, t_test, results_path, method='mf') 

sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf.csv')
sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf.csv')

mu_mf = get_posterior_predictive_mean(sample_means_mf)
lower_mf, upper_mf = get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)


# FR

sample_means_fr, sample_stds_fr = get_posterior_predictive_samples(trace_fr, 100, t_test, results_path, method='fr') 
mu_fr = get_posterior_predictive_mean(sample_means_fr)

sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr.csv')
sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr.csv')

lower_fr, upper_fr = get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,30):
      #plt.fill_between(df['year'][sep_idx:], sample_means_hmc.ix[i] - 2*sample_stds_hmc.ix[i],  sample_means_hmc.ix[i] + 2*sample_stds_hmc.ix[i], alpha=0.3, color='grey')
      plt.plot(df['year'][sep_idx:], sample_means_hmc.ix[i], color='b', alpha=0.3)


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,30):
      plt.fill_between(df['year'][sep_idx:], sample_means_mf.ix[i] - 2*sample_stds_mf.ix[i],  sample_means_mf.ix[i] + 2*sample_stds_mf.ix[i], alpha=0.3, color='grey')
      plt.plot(df['year'][sep_idx:], sample_means_mf.ix[i], color='coral', alpha=0.3)
      
plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=2)
for i in range(0,30):
     plt.fill_between(df['year'][sep_idx:], sample_means_fr.ix[i] - 2*sample_stds_fr.ix[i],  sample_means_fr.ix[i] + 2*sample_stds_fr.ix[i], alpha=0.3, color='grey')
     plt.plot(df['year'][sep_idx:], sample_means_fr.ix[i], color='green', alpha=0.3)


# Metrics

rmse_hmc = rmse(mu_hmc, y_test)
rmse_mf = rmse(mu_mf, y_test)
rmse_fr = rmse(mu_fr, y_test)

lppd_hmc, lpd_hmc = log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
lppd_mf, lpd_mf = log_predictive_mixture_density(y_test, sample_means_mf, sample_stds_mf)
lppd_fr, lpd_fr = log_predictive_mixture_density(y_test, sample_means_fr, sample_stds_fr)



# Plot with HMC + ADVI + Type II results with RMSE and LPD for co2 data

title = 'Type II ML   ' + ' RMSE: ' + str(rmse_)  + '   LPD: ' + str(lpd_) + '\n' +  ' HMC         '  + '    RMSE: ' + str(rmse_hmc) + '   LPD: ' + str(lpd_hmc) + '\n' + ' MF           ' +  '     RMSE: ' + str(rmse_mf) + '    LPD: ' + str(lpd_mf) +  '\n' + ' FR           '  +  '     RMSE: ' + str(rmse_fr)  + '    LPD: ' + str(lpd_fr) 


plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], mu_test, alpha=1, label='Type II ML', color='r')
plt.plot(df['year'][sep_idx:], mu_hmc, alpha=1, label='HMC', color='b')
plt.plot(df['year'][sep_idx:], mu_mf, alpha=1, label='MF', color='coral')
plt.plot(df['year'][sep_idx:], mu_fr, alpha=1, label='FR', color='g')

plt.fill_between(df['year'][sep_idx:], (mu_test - 1.96*std_test), (mu_test + 1.96*std_test), color='red', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], lower_fr, upper_fr, color='green', alpha=0.2)
plt.legend(fontsize='x-small')
plt.title(title, fontsize='x-small')
plt.ylim(370,420)

# Write out trace summary & autocorrelation plots

model = 'fr'
trace = trace_fr

summary_df = pm.summary(trace)
#summary_df['Acc Rate'] = np.mean(trace.get_sampler_stats('mean_tree_accept'))
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
      
      
pair_grid_plot(trace_k1, ml_deltas, k1_names, color='b')
pair_grid_plot(trace_k2, ml_deltas, k2_names, color='b')
pair_grid_plot(trace_k3, ml_deltas, k3_names, color='b')
pair_grid_plot(trace_k4, ml_deltas, k4_names, color='b')

# Pair grid catalog

from matplotlib.backends.backend_pdf import PdfPages
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
        sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=1)
        sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=1)
        sns.kdeplot(trace_hmc_load[i[0]], trace_hmc_load[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.4)
        plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
      
       

#with pm.Model() as co2_model:
#      
#    # yearly periodic component x long term trend
#    sig_var_3 = pm.HalfCauchy("sig_var_3", beta=2, testval=1.0)
#    ls_4 = pm.Gamma("ls_4", alpha=10, beta=0.1)
#    ls_5 = pm.Gamma("ls_5", alpha=4, beta=1)
#    cov_seasonal = sig_var_3*pm.gp.cov.Periodic(1, period=1, ls=ls_5) \
#                            * pm.gp.cov.ExpQuad(1, ls_4)
#    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)
#
#    # small/medium term irregularities
#    sig_var_6 = pm.HalfCauchy("sig_var_6", beta=3, testval=0.1)
#    ls_7 = pm.Gamma("ls_7", alpha=5, beta=0.75)
#    alpha_8 = pm.Gamma("alpha_8", alpha=3, beta=2)
#    cov_medium = sig_var_6*pm.gp.cov.RatQuad(1,ls_7, alpha_8)
#    gp_medium = pm.gp.Marginal(cov_func=cov_medium)
#
#    # long term trend
#    sig_var_1 = pm.HalfCauchy("sig_var_1", beta=4, testval=2.0)
#    ls_2 = pm.Gamma("ls_2", alpha=4, beta=0.1)
#    cov_trend = sig_var_1*pm.gp.cov.ExpQuad(1, ls_2)
#    gp_trend = pm.gp.Marginal(cov_func=cov_trend)
#
#    # noise model
#    sig_var_9 = pm.HalfNormal("sig_var_9", sd=0.5, testval=0.05)
#    ls_10 = pm.Gamma("ls_10", alpha=2, beta=4)
#    noise_11  = pm.HalfNormal("noise_11",  sd=0.25, testval=0.05)
#    cov_noise = sig_var_9*pm.gp.cov.ExpQuad(1, ls_10) + pm.gp.cov.WhiteNoise(noise_11)
#
#    # The Gaussian process is a sum of these three components
#    gp = gp_seasonal + gp_medium + gp_trend
#
#    # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
#    y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)
#
#with hyp_learning:
#    # this line calls an optimizer to find the MAP
#    #mp = pm.find_MAP(include_transformed=True, progressbar=True)
#    trace_hmc = pm.sample(tune=200, draws=400, chains=1)
#    
#with hyp_learning:
#    
#    pm.save_trace(trace_hmc, directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/u_prior/')
#    
#with co2_model:
#      
#    trace_hmc_load = pm.load_trace(directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
#        
#names = [name for name in mp.keys() if not name.endswith('_')] 
#mp = {}
#for i in np.arange(len(names)):
#      mp.update({names[i] : mp[names[i]]})
#      
##-------------------------------------------------------------------------------
#
## The prior model
#
##-------------------------------------------------------------------------------
#
#with pm.Model() as priors:
#      
#    sig_var_3 = pm.HalfCauchy("sig_var_3", beta=2, testval=1.0)
#    ls_4 = pm.Gamma("ls_4", alpha=10, beta=0.1)
#    ls_5 = pm.Gamma("ls_5", alpha=4, beta=1)
#    
#    sig_var_6 = pm.HalfCauchy("sig_var_6", beta=3, testval=0.1)
#    ls_7 = pm.Gamma("ls_7", alpha=5, beta=0.75)
#    alpha_8 = pm.Gamma("alpha_8", alpha=3, beta=2)
#      
#    sig_var_1 = pm.HalfCauchy("sig_var_1", beta=30, testval=2.0)
#    ls_2 = pm.Gamma("ls_2", alpha=4, beta=0.1)
#    
#    sig_var_9 = pm.HalfNormal("sig_var_9", sd=0.5, testval=0.05)
#    ls_10 = pm.Gamma("ls_10", alpha=2, beta=4)
#    noise_11  = pm.HalfNormal("noise_11",  sd=0.25, testval=0.05)
#    
#
#def plot_priors_ml(model, ml_deltas):
#      
#      plt.figure(figsize=(10,10))
#      
#      x_sv = np.linspace(0,10,100)
#      x_nv = np.linspace(0,10,100)
#      x_ls = np.linspace(0,100,1000)
#      x_a = np.linspace(0,5,100)
#      
#      plt.subplot(341)
#      plt.plot(x_sv, np.exp(model.sig_var_1.distribution.logp(x_sv).eval()))
#      plt.axvline(x=ml_deltas['sig_var_1'], color='r')
#      
#      plt.subplot(342)
#      plt.plot(x_ls, np.exp(model.ls_2.distribution.logp(x_ls).eval()))
#      plt.axvline(x=ml_deltas['ls_2'], color='r')
#      
#      plt.subplot(343)
#      plt.plot(x_sv, np.exp(model.sig_var_3.distribution.logp(x_sv).eval()))
#      plt.axvline(x=ml_deltas['sig_var_3'], color='r')
#      
#      plt.subplot(344)
#      x_ls_2 = np.linspace(0,400,1000)
#      plt.plot(x_ls_2, np.exp(model.ls_4.distribution.logp(x_ls_2).eval()))
#      plt.axvline(x=ml_deltas['ls_4'], color='r')
#      
#      plt.subplot(345)
#      plt.plot(x_ls, np.exp(model.ls_5.distribution.logp(x_ls).eval()))
#      plt.axvline(x=ml_deltas['ls_5'], color='r')
#      
#      plt.subplot(346)
#      plt.plot(x_sv, np.exp(model.sig_var_6.distribution.logp(x_sv).eval()))
#      plt.axvline(x=ml_deltas['sig_var_6'], color='r')
#      
#      plt.subplot(347)
#      plt.plot(x_ls, np.exp(model.ls_7.distribution.logp(x_ls).eval()))
#      plt.axvline(x=ml_deltas['ls_7'], color='r')
#      
#      plt.subplot(348)
#      plt.plot(x_a, np.exp(model.alpha_8.distribution.logp(x_a).eval()))
#      plt.axvline(x=ml_deltas['alpha_8'], color='r')
#      
#      plt.subplot(349)
#      plt.plot(x_sv, np.exp(model.sig_var_9.distribution.logp(x_sv).eval()))
#      plt.axvline(x=ml_deltas['sig_var_9'], color='r')
#      
#      plt.subplot(3,4,10)
#      plt.plot(x_ls[0:20], np.exp(model.ls_10.distribution.logp(x_ls[0:20]).eval()))
#      plt.axvline(x=ml_deltas['ls_10'], color='r')
#      
#      plt.subplot(3,4,11)
#      plt.plot(x_nv, np.exp(model.noise_11.distribution.logp(x_nv).eval()))
#      plt.axvline(x=ml_deltas['noise_11'], color='r')