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
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")


def normalize(y):
      
      return (y - y[0])/np.std(y)

path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/'
df = pd.read_table(path + 'mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)

# creat a date index for the data - convert properly from the decimal year 

#df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')

df.dropna(inplace=True)

first_co2 = df['co2'][0]
std_co2 = np.std(df['co2'])   

# normalize co2 levels
   
y = normalize(df['co2'])
t = df['year'] - df['year'][0]

sep_idx = 545

y_train = y[0:sep_idx].values
y_test = y[sep_idx:].values

t_train = t[0:sep_idx].values[:,None]
t_test = t[sep_idx:].values[:,None]

# sklearn kernel 

# se +  sexper + rq + se + noise

sig_var_1 = 66
lengthscale_2 = 67
sig_var_3 = 2.4
lengthscale_4 = 90 
lengthscale_5 = 1.3
sig_var_6 = 0.66
lengthscale_7 = 1.2 
alpha_8 = 0.78
sig_var_9 = 0.18
lengthscale_10 = 1.6
noise_11 = 0.19

# LT trend
sk1 = Ck(sig_var_1, constant_value_bounds=(0.1,100))*RBF(lengthscale_2, length_scale_bounds=(1,90))

# Periodic x LT Trend
sk2 = Ck(sig_var_3, constant_value_bounds=(1,5))*RBF(lengthscale_4, length_scale_bounds=(70,150))*PER(length_scale=lengthscale_5, periodicity=1, periodicity_bounds='fixed')

# Medium term irregularities
sk3 = Ck(sig_var_6, constant_value_bounds=(0.01,100))*RQ(length_scale=lengthscale_7, alpha=alpha_8, length_scale_bounds=(0.1,10), alpha_bounds=(0.2,2)) 

# Noise model
sk4 = Ck(sig_var_9, constant_value_bounds=(0.01,2))*RBF(lengthscale_10, length_scale_bounds=(0.1,2)) + WhiteKernel(noise_11)
#sk4 = WhiteKernel(noise_11)


#--------------------------------------------------------------------------------------------


# LT trend
sk1 = Ck(sig_var_1)*RBF(lengthscale_2)

# Periodic x LT Trend
sk2 = Ck(sig_var_3)*RBF(lengthscale_4)*PER(length_scale=lengthscale_5, periodicity=1, periodicity_bounds='fixed')

# Medium term irregularities
sk3 = Ck(sig_var_6)*RQ(length_scale=lengthscale_7, alpha=alpha_8, alpha_bounds=(0.1,2)) 

# Noise model
sk4 = Ck(sig_var_9)*RBF(lengthscale_10) + WhiteKernel(noise_11)
#sk4 = WhiteKernel(noise_11)
#---------------------------------------------------------------------
    
# Type II ML for hyp.
    
#---------------------------------------------------------------------
    
sk_kernel = sk1 + sk2 + sk3 + sk4
gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=10)

# Fit to data 

gpr.fit(t_train, y_train)
     
print("\nLearned kernel: %s" % gpr.kernel_)
print("Log-marginal-likelihood: %.3f"
% gpr.log_marginal_likelihood(gpr.kernel_.theta))


print("Predicting with trained gp on training data")

mu_fit_n, std_fit_n = gpr.predict(t_train, return_std=True)
mu_fit = mu_fit_n*std_co2 + first_co2
std_fit = std_fit_n*std_co2

print("Predicting with trained gp on test data")

mu_test_n, cov_test_n = gpr.predict(t_test, return_cov=True)
mu_test = mu_test_n*std_co2 + first_co2
std_test  = np.sqrt(np.diag(cov_test_n))*std_co2

rmse_ = np.round(np.sqrt(np.mean(np.square(mu_test - (y_test*std_co2 + first_co2)))), 2)

lpd = lambda x : st.norm.pdf(x, mu_test_n[0], std_test[0])
lpd_ = np.round(np.sum(lpd(y_test)),2) 

plt.figure()
plt.plot(df['year'], df['co2'], 'ko', markersize=1)
plt.plot(df['year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='grey', alpha=0.2)
plt.legend(fontsize='small')
plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')

sig_var_1 = gpr.kernel_.k1.k1.k1.k1.constant_value
sig_var_3 = gpr.kernel_.k1.k1.k2.k1.k1.constant_value
sig_var_6 = gpr.kernel_.k1.k2.k1.constant_value
sig_var_9 = gpr.kernel_.k2.k1.k1.constant_value
  
ls_2 = gpr.kernel_.k1.k1.k1.k2.length_scale
ls_4 = gpr.kernel_.k1.k1.k2.k1.k2.length_scale
ls_5 =  gpr.kernel_.k1.k1.k2.k2.length_scale
ls_7 = gpr.kernel_.k1.k2.k2.length_scale
ls_10 = gpr.kernel_.k2.k1.k2.length_scale
  
alpha_8 = gpr.kernel_.k1.k2.k2.alpha
noise_11 = gpr.kernel_.k2.k2.noise_level

ml_deltas = {'sig_var_1': sig_var_1, 'ls_2': ls_2, 'sig_var_3' : sig_var_3, 'ls_4': ls_4 , 'ls_5': ls_5 , 'sig_var_6': sig_var_6, 'ls_7': ls_7, 'alpha_8' : alpha_8, 'sig_var_9' : sig_var_9, 'ls_10' : ls_10, 'noise_11': noise_11}

ml_df = pd.DataFrame(data=ml_deltas, index=['ml'])

ml_df.to_csv('/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/co2_ml.csv', sep=',')

ml_df = pd.read_csv('/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/co2_ml.csv', index_col=None)
#-------------------------------------------------------------------------------

# The HMC model

#-------------------------------------------------------------------------------

with pm.Model() as priors:
      
    sig_var_3 = pm.HalfCauchy("sig_var_3", beta=2, testval=1.0)
    ls_4 = pm.Gamma("ls_4", alpha=10, beta=0.1)
    ls_5 = pm.Gamma("ls_5", alpha=4, beta=1)
    
    sig_var_6 = pm.HalfCauchy("sig_var_6", beta=3, testval=0.1)
    ls_7 = pm.Gamma("ls_7", alpha=5, beta=0.75)
    alpha_8 = pm.Gamma("alpha_8", alpha=3, beta=2)
      
    sig_var_1 = pm.HalfCauchy("sig_var_1", beta=10, testval=2.0)
    ls_2 = pm.Gamma("ls_2", alpha=4, beta=0.1)
    
    sig_var_9 = pm.HalfNormal("sig_var_9", sd=0.5, testval=0.05)
    ls_10 = pm.Gamma("ls_10", alpha=2, beta=4)
    noise_11  = pm.HalfNormal("noise_11",  sd=0.25, testval=0.05)
    

def plot_priors_ml(model, ml_deltas):
      
      plt.figure(figsize=(10,10))
      
      x_sv = np.linspace(0,10,100)
      x_nv = np.linspace(0,10,100)
      x_ls = np.linspace(0,100,1000)
      x_a = np.linspace(0,5,100)
      
      plt.subplot(341)
      plt.plot(x_sv, np.exp(model.sig_var_1.distribution.logp(x_sv).eval()))
      plt.axvline(x=ml_deltas['sig_var_1'], color='r')
      
      plt.subplot(342)
      plt.plot(x_ls, np.exp(model.ls_2.distribution.logp(x_ls).eval()))
      plt.axvline(x=ml_deltas['ls_2'], color='r')
      
      plt.subplot(343)
      plt.plot(x_sv, np.exp(model.sig_var_3.distribution.logp(x_sv).eval()))
      plt.axvline(x=ml_deltas['sig_var_3'], color='r')
      
      plt.subplot(344)
      x_ls_2 = np.linspace(0,400,1000)
      plt.plot(x_ls_2, np.exp(model.ls_4.distribution.logp(x_ls_2).eval()))
      plt.axvline(x=ml_deltas['ls_4'], color='r')
      
      plt.subplot(345)
      plt.plot(x_ls, np.exp(model.ls_5.distribution.logp(x_ls).eval()))
      plt.axvline(x=ml_deltas['ls_5'], color='r')
      
      plt.subplot(346)
      plt.plot(x_sv, np.exp(model.sig_var_6.distribution.logp(x_sv).eval()))
      plt.axvline(x=ml_deltas['sig_var_6'], color='r')
      
      plt.subplot(347)
      plt.plot(x_ls, np.exp(model.ls_7.distribution.logp(x_ls).eval()))
      plt.axvline(x=ml_deltas['ls_7'], color='r')
      
      plt.subplot(348)
      plt.plot(x_a, np.exp(model.alpha_8.distribution.logp(x_a).eval()))
      plt.axvline(x=ml_deltas['alpha_8'], color='r')
      
      plt.subplot(349)
      plt.plot(x_sv, np.exp(model.sig_var_9.distribution.logp(x_sv).eval()))
      plt.axvline(x=ml_deltas['sig_var_9'], color='r')
      
      plt.subplot(3,4,10)
      plt.plot(x_ls[0:20], np.exp(model.ls_10.distribution.logp(x_ls[0:20]).eval()))
      plt.axvline(x=ml_deltas['ls_10'], color='r')
      
      plt.subplot(3,4,11)
      plt.plot(x_nv, np.exp(model.noise_11.distribution.logp(x_nv).eval()))
      plt.axvline(x=ml_deltas['noise_11'], color='r')
      
      
      
      

with pm.Model() as co2_model:
      
    # yearly periodic component x long term trend
    sig_var_3 = pm.HalfCauchy("sig_var_3", beta=2, testval=1.0)
    ls_4 = pm.Gamma("ls_4", alpha=10, beta=0.1)
    ls_5 = pm.Gamma("ls_5", alpha=4, beta=1)
    cov_seasonal = sig_var_3*pm.gp.cov.Periodic(1, period=1, ls=ls_5) \
                            * pm.gp.cov.ExpQuad(1, ls_4)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

    # small/medium term irregularities
    sig_var_6 = pm.HalfCauchy("sig_var_6", beta=3, testval=0.1)
    ls_7 = pm.Gamma("ls_7", alpha=5, beta=0.75)
    alpha_8 = pm.Gamma("alpha_8", alpha=3, beta=2)
    cov_medium = sig_var_6*pm.gp.cov.RatQuad(1,ls_7, alpha_8)
    gp_medium = pm.gp.Marginal(cov_func=cov_medium)

    # long term trend
    sig_var_1 = pm.HalfCauchy("sig_var_1", beta=4, testval=2.0)
    ls_2 = pm.Gamma("ls_2", alpha=4, beta=0.1)
    cov_trend = sig_var_1*pm.gp.cov.ExpQuad(1, ls_2)
    gp_trend = pm.gp.Marginal(cov_func=cov_trend)

    # noise model
    sig_var_9 = pm.HalfNormal("sig_var_9", sd=0.5, testval=0.05)
    ls_10 = pm.Gamma("ls_10", alpha=2, beta=4)
    noise_11  = pm.HalfNormal("noise_11",  sd=0.25, testval=0.05)
    cov_noise = sig_var_9*pm.gp.cov.ExpQuad(1, ls_10) + pm.gp.cov.WhiteNoise(noise_11)

    # The Gaussian process is a sum of these three components
    gp = gp_seasonal + gp_medium + gp_trend

    # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
    y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)

with co2_model:
    # this line calls an optimizer to find the MAP
    #mp = pm.find_MAP(include_transformed=True, progressbar=True)
    trace_hmc = pm.sample(tune=200, draws=400, chains=1)
    
with co2_model:
    
    pm.save_trace(trace_hmc, directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
    
with co2_model:
      
    trace_hmc_load = pm.load_trace(directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
        
names = [name for name in mp.keys() if not name.endswith('_')] 
mp = {}
for i in np.arange(len(names)):
      mp.update({names[i] : mp[names[i]]})

varnames = ['ls_2','ls_4','ls_5','ls_7','ls_10','sig_var_1','sig_var_3','sig_var_6','sig_var_9','alpha_8','noise_11']  

def get_trace_df(trace_hmc, varnames):
      
    trace_df = pd.DataFrame()
    for i in varnames:
          trace_df[i] = trace_hmc.get_values(i)
    return trace_hmc

trace_df2 = get_trace_df(trace_hmc_load, varnames)


traces_part1 = pm.traceplot(trace_hmc_load, varnames[0:5], lines=ml_deltas)
traces_part2 = pm.traceplot(trace_hmc_load, varnames[5:], lines=ml_deltas)

for i in np.arange(5):
      
      delta = deltas.get(str(varnames[i]))
      xmin=0
      xmax = max(max(trace_hmc[varnames[i]]), delta)
      traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
      traces_part1[i][0].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
      traces_part1[i][0].axes.set_xlim(xmin, xmax)
      traces_part1[i][0].legend(fontsize='x-small')

for i in np.arange(6):
      
      delta = deltas.get(str(varnames[i+5]))
      traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
      traces_part2[i][0].hist(trace_hmc[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
      traces_part2[i][0].axes.set_xscale('log')
      traces_part2[i][0].legend(fontsize='x-small')
      

def get_posterior_predictive_gp_trace(trace, thin_factor, X_star):
      
      #sq_arr = np.empty(shape=(l,))
      means_writer = csv.writer(open('means_hmc.csv','w')) 
      std_writer = csv.writer(open('std_hmc.csv','w'))
      sq_writer = csv.writer(open('sq_hmc.csv','w'))
      
      means_writer.writerow(df['year'][sep_idx:])
      std_writer.writerow(df['year'][sep_idx:])
      sq_writer.writerow(df['year'][sep_idx:])
      
      for i in np.arange(len(trace))[::thin_factor]:
            
            print(i)
            mu, var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            std = np.sqrt(var)
            
            means_writer.writerow(mu)
            std_writer.writerow(std)
            sq_writer.writerow(np.multiply(mu, mu))

      sample_means =  pd.read_csv(path+ 'means_hmc.csv', sep=',', header=0)
      rescaled_means = (sample_means)*std_co2 + first_co2
      
      sample_stds = pd.read_csv(path+ 'std_hmc.csv', sep=',', header=0)
      sample_sqs = pd.read_csv(path+ 'sq_hmc.csv')
      
      mu_hmc = np.mean(sample_means)*std_co2 + first_co2
      sd_hmc = (np.mean(sample_stds) + np.mean(sample_sqs)  - np.multiply(np.mean(sample_means), np.mean(sample_means)))*std_co2
      
      return mu_hmc, sd_hmc, rescaled_means, sample_stds*std_co2

mu_hmc, sd_hm, sample_means, sample_stds = get_posterior_predictive_gp_trace(trace_hmc_load, 5, t_test) 
mu_hmc1, sd_hm1, sample_means1, sample_stds1 = get_posterior_predictive_gp_trace(trace_hmc_load, 20, t_test) 




# Plot with HMC results

plt.figure()
plt.plot(df['year'][sep_idx:], df['co2'][sep_idx:], 'ko', markersize=1)
plt.plot(df['year'][sep_idx:], sample_means1.T, color='grey', alpha=0.3)
plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='Type II ML', color='r')
plt.plot(df['year'][sep_idx:], mu_hmc1, alpha=0.5, label='HMC', color='b')
plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='red', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], mu_hmc1 - 2*sd_hm1, mu_hmc1 + 2*sd_hm1, color='blue', alpha=0.2)
plt.legend(fontsize='x-small')

# Write out trace summary & autocorrelation plots

prefix = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Config/'
summary_df = pm.summary(trace_hmc)
summary_df['Acc Rate'] = np.mean(trace_hmc.get_sampler_stats('mean_tree_accept'))
np.round(summary_df,3).to_csv(prefix + 'trace_summary_co2.csv')
      
      


























     #-----------------------------------------------------
      
     #       Hybrid Monte Carlo
          
     #-----------------------------------------------------

     with pm.Model() as hyp_learning:
        
             # prior on lengthscales
             
             log_l2 = pm.Uniform('log_l2', lower=-5, upper=6)
             log_l4 = pm.Uniform('log_l4', lower=-5, upper=6)
             log_l5 = pm.Uniform('log_l5', lower=-5, upper=6)
             log_l7 = pm.Uniform('log_l7', lower=-5, upper=6)
             log_l10 = pm.Uniform('log_l10', lower=-5, upper=6)

             ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
             ls_4 = pm.Deterministic('ls_4', tt.exp(log_l4))
             ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
             ls_7 = pm.Deterministic('ls_7', tt.exp(log_l7))
             ls_10 = pm.Deterministic('ls_10', tt.exp(log_l10))
             
             # prior on amplitudes
             
             log_sv1 = pm.Uniform('log_sv1', lower=-10, upper=10)
             log_sv3 = pm.Uniform('log_sv3', lower=-10, upper=10)
             log_sv6 = pm.Uniform('log_sv6', lower=-10, upper=10)
             log_sv9 = pm.Uniform('log_sv9', lower=-10, upper=10)

             sig_var_1 = pm.Deterministic('sig_var_1', tt.exp(log_sv1))
             sig_var_3 = pm.Deterministic('sig_var_3', tt.exp(log_sv3))
             sig_var_6 = pm.Deterministic('sig_var_6', tt.exp(log_sv6))
             sig_var_9 = pm.Deterministic('sig_var_9', tt.exp(log_sv9))

             # prior on alpha
            
             log_alpha8 = pm.Uniform('log_alpha8', lower=-10, upper=10)
             alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
             
             # prior on noise variance term
            
             log_nv11 = pm.Uniform('log_nv11', lower=-5, upper=5)
             noise_11 = pm.Deterministic('noise_11', tt.exp(log_nv11))
               
             
             # Specify the covariance function
             
             k1 = pm.gp.cov.Constant(sig_var_1)*pm.gp.cov.ExpQuad(1, ls_2) 
             k2 = pm.gp.cov.Constant(sig_var_3)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
             k3 = pm.gp.cov.Constant(sig_var_6)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
             k4 = pm.gp.cov.Constant(sig_var_9)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(noise_11)
      
             k = k1 + k2 + k3 + k4
                
             gp = pm.gp.Marginal(cov_func=k)
                  
             # Marginal Likelihood
             y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=np.sqrt(noise_11))
             
             # HMC Nuts auto-tuning implementation
             trace_hmc = pm.sample(chains=1, start=ml_deltas)
             
#      with hyp_learning:
#            
#              # this line calls an optimizer to find the MAP
#              #mp = pm.find_MAP(include_transformed=True)
#               fnew = gp.conditional("fnew", Xnew=t_test)
#                  
#      
#      with hyp_learning:
#            
#             advi = pm.FullRankADVI()
#             advi.fit()    
#             trace_advi = advi.approx.sample(include_transformed=False) 
      