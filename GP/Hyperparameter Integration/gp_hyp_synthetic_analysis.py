#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:48:56 2019

@author: vidhi

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


#------------------------Data Generation----------------------------------------------------------

def generate_gp_latent(X_star, mean, cov):
    
    return np.random.multivariate_normal(mean(X_star).eval(), cov=cov(X_star, X_star).eval())

def generate_gp_training(X_all, f_all, n_train, noise_var, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.5*st.norm.pdf(X_all, 2, 0.7) + 0.5*st.norm.pdf(X_all, 15.0,1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
     
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    X = X_all[train_index]
    f = f_all[train_index]
    X_star = X_all[test_index]
    f_star = f_all[test_index]
    y = f + np.random.normal(0, scale=np.sqrt(noise_var), size=n_train)
    return X, y, X_star, f_star, f

def generate_fixed_domain_training_sets(X_all, f_all, noise_var, uniform, seq_n_train):
      
      data_sets = {}
      for i in seq_n_train:
            X, y, X_star, f_star, f = generate_gp_training(X_all, f_all, i, noise_var, uniform)
            data_sets.update({'X_' + str(i): X})
            data_sets.update({'y_' + str(i): y})
            data_sets.update({'X_star_' + str(i): X_star})
            data_sets.update({'f_star_' + str(i): f_star})
      return data_sets
            
def persist_datasets(X, y, X_star, f_star, path, suffix):
      
     X.tofile(path + 'X' + suffix + '.csv', sep=',')
     X_star.tofile(path + 'X_star' + suffix + '.csv', sep=',')
     y.tofile(path + 'y' +  suffix + '.csv', sep=',')
     f_star.tofile(path + 'f_star' + suffix + '.csv', sep=',')

#----------------------------Loading persisted data----------------------------

def load_datasets(path, n_train):
      
       n_test = 200 - n_train
       X = np.asarray(pd.read_csv(path + 'X_' + str(n_train) + '.csv', header=None)).reshape(n_train,1)
       y = np.asarray(pd.read_csv(path + 'y_' + str(n_train) + '.csv', header=None)).reshape(n_train,)
       X_star = np.asarray(pd.read_csv(path + 'X_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,1)
       f_star = np.asarray(pd.read_csv(path + 'f_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,)
       return X, y, X_star, f_star

#----------------------------GP Inference-------------------------------------
    
# Type II ML for hyp.
    
    
def get_ml_report(X, y, X_star, f_star):
      
          kernel = Ck(10.0, (1e-10, 1e2)) * RBF(2, length_scale_bounds=(0.5, 8)) + WhiteKernel(10.0, noise_level_bounds=(1e-5,100))
          
          gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
              
          # Fit to data 
          gpr.fit(X, y)        
          post_mean, post_cov = gpr.predict(X_star, return_cov = True) 
          post_std = np.sqrt(np.diag(post_cov))
          post_samples = np.random.multivariate_normal(post_mean, post_cov, 10)
          rmse_ = rmse(post_mean, f_star)
          lpd_ = log_predictive_density(f_star, post_mean, post_cov)
          title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_)     
          ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
          ml_deltas_dict = {'lengthscale': ml_deltas[1], 'noise_var': ml_deltas[2], 'sig_var': ml_deltas[0]}
          #plot_lml_surface_3way(gpr, ml_deltas_dict['sig_var'], ml_deltas_dict['lengthscale'], ml_deltas_dict['noise_var'])
          return post_mean, post_std, rmse_, lpd_, ml_deltas_dict, title

@sampled
def generative_model(X, y):
      
       # prior on lengthscale 
       log_l = pm.Uniform('log_l', lower=-5, upper=5)
       lengthscale = pm.Deterministic('lengthscale', tt.exp(log_l))
       
        #prior on noise variance
       log_nv = pm.Uniform('log_nv', lower=-5, upper=5)
       noise_var = pm.Deterministic('noise_var', tt.exp(log_nv))
         
       #prior on signal variance
       log_sv = pm.Uniform('log_sv', lower=-10, upper=5)
       sig_var = pm.Deterministic('sig_var', tt.exp(log_sv))
       
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
                   
  
#--------------------Predictive performance metrics-------------------------
      
def rmse(post_mean, f_star):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - f_star))),3)

def log_predictive_density(f_star, post_mean, post_cov):

      return np.round(np.sum(np.log(st.multivariate_normal.pdf(f_star, post_mean, post_cov, allow_singular=True))), 3)

def log_predictive_mixture_density(f_star, list_means, list_cov):
      
      components = []
      for i in np.arange(len(list_means)):
            components.append(st.multivariate_normal.pdf(f_star, list_means[i].eval(), list_cov[i].eval(), allow_singular=True))
      return np.round(np.sum(np.log(np.mean(components))),3)


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
    
def plot_gp_ml_II_joint(X, y, X_star, pred_mean, pred_std, title):
      
      plt.figure(figsize=(20,5))
      
      for i in [0, 1,2,3]:
            plt.subplot(1,4,i+1)
            plt.plot(X_star[i], pred_mean[i], color='r')
            plt.plot(X_star[i], f_star[i], 'k', linestyle='dashed')
            plt.plot(X[i], y[i], 'ko', markersize=2)
            plt.fill_between(X_star[i].ravel(), pred_mean[i] -1.96*pred_std[i], pred_mean[i] + 1.96*pred_std[i], color='r', alpha=0.3)
            plt.title(title[i], fontsize='small')
      plt.suptitle('Type II ML')
      

      plt.figure()
      plt.plot(n_train, [rmse_ml_10, rmse_ml_20, rmse_ml_40, rmse_ml_60])
      plt.xlabel('N train')
      plt.ylabel('RMSE')
 

if __name__ == "__main__":


      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/1-d/'
      home_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/1-d/'

      
      input_dist =  'Unif'
      snr = 10
      n_train = [10, 20, 40, 60]
      path = uni_path + input_dist + '/' + 'SNR_' + str(snr) + '/Training/' 

      
      #----------------------------------------------------
      # Joint Analysis
      #----------------------------------------------------
      
      X_10, y_10, X_star_10, f_star_10 = load_datasets(path, 10)
      X_20, y_20, X_star_20, f_star_20 = load_datasets(path, 20)
      X_40, y_40, X_star_40, f_star_40 = load_datasets(path, 40)
      X_60, y_60, X_star_60, f_star_60 = load_datasets(path, 60)
      
      X = [X_10, X_20, X_40, X_60]
      y = [y_10, y_20, y_40, y_60]
      X_star = [X_star_10, X_star_20, X_star_40, X_star_60]
      f_star = [f_star_10, f_star_20, f_star_40, f_star_60] 
      
      # Collecting ML stats for 1 generative model
      
      pp_mean_ml_10, pp_std_ml_10, rmse_ml_10, lpd_ml_10, ml_deltas_dict_10, title_10 =  get_ml_report(X_10, y_10, X_star_10, f_star_10)
      pp_mean_ml_20, pp_std_ml_20, rmse_ml_20, lpd_ml_20, ml_deltas_dict_20, title_20 =  get_ml_report(X_20, y_20, X_star_20, f_star_20)
      pp_mean_ml_40, pp_std_ml_40, rmse_ml_40, lpd_ml_40, ml_deltas_dict_40, title_40 =  get_ml_report(X_40, y_40, X_star_40, f_star_40)
      pp_mean_ml_60, pp_std_ml_60, rmse_ml_60, lpd_ml_60, ml_deltas_dict_60, title_60 =  get_ml_report(X_60, y_60, X_star_60, f_star_60)

      title = [title_10, title_20, title_40, title_60]
      pred_mean = [pp_mean_ml_10, pp_mean_ml_20, pp_mean_ml_40, pp_mean_ml_60] 
      pred_std = [pp_std_ml_10, pp_std_ml_20, pp_std_ml_40, pp_std_ml_60]



# TODO Clean up from here

# Collecting HMC stats for 1 generative model
model_10 = get_pymc3_model(X_10, y_10)
model_20 = get_pymc3_model(X_20, y_20)
model_40 = get_pymc3_model(X_40, y_40)
model_60 = get_pymc3_model(X_60, y_60)

trace_hmc_10 = get_hmc_report(get_pymc3_model(X_10, y_10), X_star_10)
trace_hmc_20 =  get_hmc_report(get_pymc3_model(X_20, y_20), X_star_20)
trace_hmc_40 =  get_hmc_report(get_pymc3_model(X_40, y_40), X_star_40)
trace_hmc_60 =  get_hmc_report(get_pymc3_model(X_60, y_60), X_star_60)

# Persist traces

with model_10:
      



pp_mean_hmc_10, pp_std_hmc_10, means_10, std_10, sq_10 = get_posterior_predictive_gp_trace(trace_hmc_10, 10, X_star_10, path, 10)
pp_mean_hmc_20, pp_std_hmc_20, means_20, std_20, sq_20 = get_posterior_predictive_gp_trace(trace_hmc_20, 10, X_star_20, path, 20)
pp_mean_hmc_40, pp_std_hmc_40, means_40, std_40, sq_40 = get_posterior_predictive_gp_trace(trace_hmc_40, 10, X_star_40, path, 40)
pp_mean_hmc_60, pp_std_hmc_60, means_60, std_60, sq_60 = get_posterior_predictive_gp_trace(trace_hmc_60, 10, X_star_60, path, 60)


plt.figure(figsize=(20,5))
plt.plot(X_star_10, np.mean(pp_mean_hmc_10['f_pred'], axis=0), 'b', alpha=0.4)
plt.plot(X_star_20, np.mean(pp_mean_hmc_20['f_pred'], axis=0), 'b', alpha=0.4)
plt.plot(X_star_40, np.mean(pp_mean_hmc_40['f_pred'], axis=0), 'b', alpha=0.4)
plt.plot(X_star_60, np.mean(pp_mean_hmc_60['f_pred'], axis=0), 'b', alpha=0.4)

plt.plot(X_star_10, f_star_10, 'k', linestyle='dashed')

plt.plot(X_star_10, pp_mean_hmc_10['f_pred'].T, 'b', alpha=0.4)

# Collecting ADVI stats for 1 generative model



from scipy.stats import gaussian_kde

# Handling HMC traces for N_train
      
traces = pm.traceplot(trace_hmc_60, varnames, combined=True) 

for i in [0,1,2]:
      
      density_40 = gaussian_kde(trace_hmc_40.get_values(varnames[i]))
      density_20 = gaussian_kde(trace_hmc_20.get_values(varnames[i]))
      density_10 = gaussian_kde(trace_hmc_10.get_values(varnames[i]))

      traces[0][i].plot(np.linspace(0,140,1000), density_40(np.linspace(0,140, 1000)))
      traces[0][i].plot(np.linspace(0,140,1000), density_20(np.linspace(0,140, 1000)))
      traces[0][i].plot(np.linspace(0,140,1000), density_10(np.linspace(0,140, 1000)))

      
# Handling HMC traces for SNR
      

# Hamdling HMC traces for Unif / NUnif