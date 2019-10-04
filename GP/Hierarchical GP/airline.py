#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:32:30 2019

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
import autograd.numpy as np
from autograd import elementwise_grad, jacobian, grad
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa
import advi_analysis as ad

# Analytical variational inference for Airline data

def kernel(theta, X1, X2):
        
     # se +  sexper + rq + se 
     
     s_1 = theta[0]
     ls_2 = theta[1]
     ls_3 = theta[2]
     
     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
     dist = np.abs(np.sum(X1,1).reshape(-1,1) - np.sum(X2,1))
     sk = s_1**2 * np.exp(-0.5 / ls_2**2 * sqdist) * np.exp(-2*(np.sin(np.pi*dist/366)/ls_3)**2)
    
     return sk

def gp_mean(theta, X, y, X_star):
      
     n_4 = theta[3]
     n_1 = theta[4]
     n_2 = theta[5]
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk2 = n_1**2 * np.exp(-0.5 / n_2**2 * sqdist_X) + n_4**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk2*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

def gp_cov(theta, X, y, X_star):
      
     n_4 = theta[3]
     n_1 = theta[4]
     n_2 = theta[5]
  
     #sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_star**2, 1) - 2 * np.dot(X, X_star.T)
     sqdist_X =  np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
     sk2 = n_1**2 * np.exp(-0.5 / n_2**2 * sqdist_X) + n_4**2*np.eye(len(X))
      
     K = kernel(theta, X, X)
     K_noise = K + sk2*np.eye(len(X))
     K_s = kernel(theta, X, X_star)
     K_ss = kernel(theta, X_star, X_star)
     return K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), K_s)
            
   
dh = elementwise_grad(gp_mean)
d2h = jacobian(dh)
dg = grad(gp_cov)
d2g = jacobian(dg) 

def get_vi_analytical(X, y, X_star, dh, d2h, d2g, theta, mu_theta, cov_theta, results_path):
                  
    #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), theta)      
    #pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
    #pred_vi_var =  np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))
    
    pred_g_mean = gp_mean(theta, X, y, X_star)
    pred_g_var = np.diag(gp_cov(theta, X, y, X_star))

    pred_ng_mean = []
    pred_ng_var = []
    
    # To fix this 
    
    #pred_ng_mean = pred_g_mean + 0.5*np.trace(np.matmul(d2h(theta, X, y, X_star), np.array(cov_theta)))
    #pred_ng_var = pred_vi_var + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta))

    for i in np.arange(len(X_star)): # To vectorize this loop
          
          print(i)
          x_star = X_star[i].reshape(1,1)

          pred_ng_mean.append(pred_g_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta))))
          pred_ng_var.append(pred_g_var[i] + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta)))

    np.savetxt(fname=results_path + 'pred_dist/' + 'mu_taylor.csv', X=pred_ng_mean, delimiter=',', header='')   
    np.savetxt(fname=results_path + 'pred_dist/' + 'std_taylor.csv', X=np.sqrt(pred_ng_var), delimiter=',', header='')   

    return pred_ng_mean, np.sqrt(pred_ng_var)


mu_taylor, std_taylor = get_vi_analytical(t_train, y_train, t_test, dh, d2h, d2g, theta, mu_theta, cov_theta, results_path)
plt.plot(t_test, mu_taylor)

def forward_mu(x, mu, std):
      
      return x*std + mu

def forward_std(x, std):
      
      return x*std


if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Airline/'
      
      #results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'
      results_path = '/Users/vidhi.lalchand/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = home_path
      
      df = pd.read_csv(path + 'AirPassengers.csv', infer_datetime_format=True, parse_dates=True, na_values=-99.99, keep_default_na=False, dtype = {'Month': np.str,'Passengers': np.int})
      
      dates = []
      for x in df['Month']:
            dates.append(np.datetime64(x))
      
      df['Time_int'] = dates - dates[0]
      df['Time_int'] = df['Time_int'].astype('timedelta64[D]')
     
      df['Year'] = pd.to_datetime(df['Month'])
      
      ctime = lambda x: (float(x.strftime("%j"))-1) / 366 + float(x.strftime("%Y"))
      df['Year'] = df['Year'].apply(ctime)
      
      emp_mu = np.mean(y)
      emp_std = np.std(y)
           
      y = df['Passengers']
      t = df['Time_int']
      
      # Whiten the data 
     
      #y =  (y - emp_mu)/emp_sd
      
      sep_idx = 100
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      
      # ML 
      
#      import GPy as gp
#
#      k1 = gp.kern.RBF(input_dim=1, variance=1, lengthscale=1, active_dims=1)
#      k2 = gp.kern.StdPeriodic(input_dim=1, variance=1, period=365, lengthscale=1,active_dims=1)
#      #k = gp.kern.Prod(k1,k2, input_dim=1)
#      m =  gp.models.GPRegression(t_train, y_train.reshape(len(y_train),1), kernel = k1*k2)
#      
#      # We will use the simplest form of GP model, exact inference
#      class ExactGPModel(gpytorch.models.ExactGP):
#          def __init__(self, train_x, train_y, likelihood):
#              super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#              self.mean_module = gpytorch.means.LinearMean(input_size=1)
#              self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ProductKernel(gpytorch.kernels.RBFKernel(), gpytorch.kernels.PeriodicKernel()))
#      
#          def forward(self, x):
#              mean_x = self.mean_module(x)
#              covar_x = self.covar_module(x)
#              return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#      
#      data=torch.tensor(np.array(df[['Passengers', 'Time_int']]))
#      train_x = data[:,:-1].double()
#      train_y = data[:,-1].double()
#      
#      # initialize likelihood and model
#      likelihood = gpytorch.likelihoods.GaussianLikelihood()
#      model = ExactGPModel(train_x, train_y, likelihood).double()
#      
#      # Find optimal model hyperparameters
#      model.train()
#      likelihood.train()
#      
#      # Use the adam optimizer
#      optimizer = torch.optim.Adam([
#          {'params': model.parameters()},  # Includes GaussianLikelihood parameters
#      ], lr=0.1)
#      
#      # "Loss" for GPs - the marginal log likelihood
#      mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#      
#      training_iter = 50
#      for i in range(training_iter):
#          # Zero gradients from previous iteration
#          optimizer.zero_grad()
#          # Output from model
#          output = model(train_x)
#          # Calc loss and backprop gradients
#          loss = -mll(output,train_y)
#          loss.backward()
#          print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#              i + 1, training_iter, loss.item(),
#              model.covar_module.lengthscale.item(),
#              model.likelihood.noise.item()
#          ))
#          optimizer.step()
      
      
      # sklearn kernel 
      
      # se +  sexper + noise
      
      #sk1 = 50**2 * Matern(length_scale=1.0, length_scale_bounds=(0.000001,1.0))
      sk2 = 20**2 * RBF(length_scale=1.0, length_scale_bounds=(1e-3,1e+7)) \
          * PER(length_scale=1.0, periodicity=365.0, periodicity_bounds='fixed')  # seasonal component
          
      #sk4 = 0.1**2 * RBF(length_scale=0.1, length_scale_bounds=(1e-3,1e3)) + WhiteKernel(noise_level=0.1**2,
                      #  noise_level_bounds=(1e-3, 100))  # noise terms
      sk_noise = WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, 100))  # noise terms
      
      #---------------------------------------------------------------------
          
      # Type II ML for hyp.
          
      #---------------------------------------------------------------------
      
      varnames = ['s_1', 'ls_2', 'ls_3', 'n_4']
      
      sk_kernel =  sk2 + sk_noise
      
      # Fit to data 
#      for i in np.arange(9):
#            gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=50)
#            print('Fitting ' + str(i))
#            gpr.fit(t_train, y_train)
#            gpr_models.append(gpr)
#            gpr_lml.append(gpr.log_marginal_likelihood(gpr.kernel_.theta))
#            gpr_ml_deltas.append(gpr.kernel_)
      
      gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=100)
      gpr.fit(t_train, y_train)
      
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
      mu_fit, std_fit = gpr.predict(t_train, return_std=True)      
      mu_test, std_test = gpr.predict(t_test, return_std=True)
      
      #mu_fit = forward_mu(mu_fit, emp_mu, emp_std)
      #std_fit = forward_std(std_fit, emp_std)
      
      #mu_test = forward_mu(mu_test, emp_mu, emp_std)
      #std_test = forward_std(std_test, emp_std)
      
      rmse_ = pa.rmse(mu_test, y_test)
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      se_rmse = pa.se_of_rmse(mu_test, y_test)
      std_lpd = pa.se_of_lpd(y_test, mu_test, std_test)
      
      print('rmse_ml: ' + str(rmse_))
      print('se_rmse_ml: ' + str(se_rmse))
      print('lpd_:' + str(lpd_))
      print('se_lpd:' + str(std_lpd))
   
      
      #Plotting
      
      plt.figure()
      plt.plot(df['Year'], df['Passengers'], 'ko', markersize=2)
      plt.plot(df['Year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'NLPD: ' + str(-lpd_), fontsize='small')
      
      s_1 = np.sqrt(gpr.kernel_.k1.k1.k1.constant_value)
      ls_2 = gpr.kernel_.k1.k1.k2.length_scale
      
      ls_3 = gpr.kernel_.k1.k2.length_scale
      
      n_4 = np.sqrt(gpr.kernel_.k2.noise_level)
            
      ml_deltas = {'s_1': s_1, 'ls_2': ls_2, 'ls_3' : ls_3, 'n_4': n_4}
      
      ml_values = [ml_deltas[v] for v in varnames]
      
      ml_df = pd.DataFrame(data=np.column_stack((varnames, ml_values)), columns=['hyp','values'])
      
      ml_df.to_csv(results_path + 'airline_ml.csv', sep=',')
      
      # Reading what is already stored
      
      ml_df = pd.read_csv(results_path + 'airline_ml.csv')
      
      ml_deltas = dict(zip(ml_df['hyp'], ml_df['values']))
       
     #-----------------------------------------------------

     #       Hybrid Monte Carlo + ADVI Inference 
    
     #-----------------------------------------------------
     
with pm.Model() as airline_model:
      
       #i = pm.Normal('i', sd=1)
       #c = pm.HalfNormal('c', sd=2.0)
       #mean_trend = pm.gp.mean.Linear(coeffs=c, intercept=i)
      
       log_l2 = pm.Normal('log_l2', mu=0, sd=2)
       log_l3 = pm.Normal('log_l3', mu=0, sd=2)
       #log_l5 = pm.Normal('log_l5', mu=np.log(ml_deltas['ls_5']), sd=2)

       ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
       ls_3 = pm.Deterministic('ls_3', tt.exp(log_l3))
       #ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
       
       p = 366
       
       # prior on amplitudes

       #log_s1 = pm.Normal('log_s1', mu=np.log(ml_deltas['s_1']), sd=2)
       #s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_1 = pm.Gamma('s_1', alpha=60, beta=0.17)       
       log_n4 = pm.Normal('log_n4', mu=np.log(ml_deltas['n_4']), sd=2)
       
       
       #log_s4 = pm.Normal('log_s4', mu=np.log(ml_deltas['s_4']), sd=2)

       n_4 = pm.Deterministic('n_4', tt.exp(log_n4))
       
       n_1 = pm.Gamma('n_1',2,0.1)
       n_2 = pm.Gamma('n_2', 2, 0.1)
                 
       # Specify the covariance function
       
       cov_main = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2)*pm.gp.cov.Periodic(1, period=p, ls=ls_3)
       cov_noise = pm.gp.cov.Constant(n_1**2)*pm.gp.cov.ExpQuad(1, n_2) + pm.gp.cov.WhiteNoise(n_4**2)
       
       gp_main = pm.gp.Marginal(cov_func=cov_main)
       gp_noise = pm.gp.Marginal(cov_func=cov_noise)

       #k = k2 
          
       gp = gp_main
       
       trace_prior = pm.sample(500)

with airline_model:
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)
       #prior_pred = pm.sample_ppc(trace_prior, samples=50)
       
with airline_model:
      
       trace_hmc = pm.sample(draws=500, tune=500, chains=2)


with airline_model:
    
      pm.save_trace(trace_hmc, directory = results_path + 'Traces_pickle_hmc_final/', overwrite=True)
      
with airline_model:
      
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/')
      

rv_mapping = {'s_1':  airline_model.log_s1, 
              'ls_2': airline_model.log_l2, 
              'ls_3':  airline_model.log_l3,
              's_4': airline_model.log_s4,
              'ls_5': airline_model.log_l5,
              'n_6': airline_model.log_n6}
          

raw_mapping = {'log_s1':  airline_model.log_s1, 
              'log_ls2': airline_model.log_l2, 
              'log_ls3':  airline_model.log_ls3,
              'log_s4': airline_model.log_l4,
              'log_ls5': airline_model.log_l5,
              'log_n6': airline_model.log_n6}
              
varnames = ['s_1','ls_2','ls_3', 'n_4', 'n_1', 'n_2']

name_mapping = {'s_1_log__':  's_1', 
              'log_l2': 'ls_2', 
              'log_l3':  'ls_3',
              'n_1_log__': 'n_1',
              'n_2_log__': 'n_2',
              'log_n4': 'n_4'}
        
with airline_model:
      
      mf = pm.ADVI()

      tracker_mf = pm.callbacks.Tracker(
      mean = mf.approx.mean.eval,    
      std = mf.approx.std.eval)
     
      mf.fit(n=25000, callbacks=[tracker_mf])
      
      trace_mf = mf.approx.sample(500)
      
with airline_model:
      
      fr = pm.FullRankADVI()
        
      tracker_fr = pm.callbacks.Tracker(
      mean = fr.approx.mean.eval,    
      std = fr.approx.std.eval)
      
      fr.fit(n=70000, callbacks=[tracker_fr])
      trace_fr = fr.approx.sample(500)
      
      
      bij_mf = mf.approx.groups[0].bij
      mf_param = {param.name: bij_mf.rmap(param.eval())
	  for param in mf.approx.params}

      bij_fr = fr.approx.groups[0].bij
      fr_param = {param.name: bij_fr.rmap(param.eval())
	  for param in fr.approx.params}

      # Updating with implicit values - %TODO Testing
      
      mf_param = ad.analytical_variational_opt(airline_model, mf_param, pm.summary(trace_mf), name_mapping)
      fr_param = ad.analytical_variational_opt(airline_model, fr_param, pm.summary(trace_fr), name_mapping)

      # Saving raw ADVI results
      
      mf_df = pd.DataFrame(mf_param)
      fr_df = pd.DataFrame(fr_param)
      
      mu_theta = fr_df['mu_implicit'][varnames] 
      cov_theta = pa.get_empirical_covariance(trace_fr_df, varnames)
      
       #Save df 
      
       trace_mf_df.to_csv(results_path + '/trace_mf_df.csv', sep=',')
       trace_fr_df.to_csv(results_path + '/trace_fr_df.csv', sep=',')

   
      
      # Loading persisted trace
   
      trace_hmc_load = pm.load_trace(results_path + 'Traces_pickle_hmc/', model=airline_model)
      
      trace_hmc_df = pm.trace_to_dataframe(trace_hmc)
      trace_mf_df = pm.trace_to_dataframe(trace_mf)
      trace_fr_df = pm.trace_to_dataframe(trace_fr)

      # Traceplots

      pa.traceplots(trace_hmc, varnames, ml_deltas, 4, combined=False, clr='b')
      pa.traceplots(trace_mf, varnames, ml_deltas, 3, True, clr='coral')
      pa.traceplots(trace_fr, v_sub, ml_deltas, 5, True, clr='g')
      
      # Traceplots compare
      
      pa.traceplot_compare(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, ml_deltas, rv_mapping, 6)
      plt.suptitle('Marginal Hyperparameter Posteriors', fontsize='small')
     
      # Prior Posterior Plot
      
      pa.plot_prior_posterior_plots(trace_prior, trace_hmc, varnames, ml_deltas, 'Prior Posterior HMC')
      pa.plot_prior_posterior_plots(trace_prior, trace_mf, varnames, ml_deltas, 'Prior Posterior MF')
      pa.plot_prior_posterior_plots(trace_prior, trace_fr, varnames, ml_deltas, 'Prior Posterior FR')
      
      pa.traceplots_two_way_compare(trace_mf, trace_fr, varnames, ml_deltas, 'Posteriors MF / FR', 'MF', 'FR')

      # Autocorrelations
      
      pm.autocorrplot(trace_hmc, varnames)
      
      # Saving summary stats 
      
      hmc_summary_df = pm.summary(trace_hmc).ix[varnames]
      hmc_summary_df.to_csv(results_path + '/hmc_summary_df.csv', sep=',')

      # Pair Grid plots 
      
      trace = trace_mf
      clr='coral'
      
      k1_names = ['s_1', 'ls_2', 'ls_3']
      k2_names = ['s_4', 'ls_5', 'n_6']
      
      trace_k1 = pa.get_subset_trace(trace, k1_names)
      trace_k2 = pa.get_subset_trace(trace, k2_names)
      
      pa.pair_grid_plot(trace_k1, ml_deltas, k1_names, color=clr)
      pa.pair_grid_plot(trace_k2, ml_deltas, k2_names, color=clr)
      
      
      # Pair scatter plot 
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
        sns.kdeplot(trace_fr[i[0]], trace_fr[i[1]], color='g', shade=True, bw='silverman', shade_lowest=False, alpha=0.9)
        #sns.kdeplot(trace_hmc[i[0]], trace_hmc[i[1]], color='b', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        sns.kdeplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', shade=True, bw='silverman', shade_lowest=False, alpha=0.8)
        #sns.scatterplot(trace_mf[i[0]], trace_mf[i[1]], color='coral', size=1, legend=False)
        #sns.scatterplot(trace_fr[i[0]], trace_fr[i[1]], color='g', size=1, legend=False)
        plt.scatter(ml_deltas[i[0]], ml_deltas[i[1]], marker='x', color='r')
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.tight_layout()
      
      
      # Testing convergence of ADVI - TODO 
      
      ad.convergence_report(tracker_mf,  mf.hist, varnames,  'Mean Field Convergence Report')
      ad.convergence_report(tracker_fr, fr.hist, varnames, 'Full Rank Convergence Report')
            
      # Predictions

      # HMC
      
      pa.write_posterior_predictive_samples(trace_hmc, 10, t_test, results_path + 'pred_dist/', method='hmc_final3', gp=gp) 
      
      sample_means_hmc = pd.read_csv(results_path + 'pred_dist/' + 'means_hmc_final3.csv')
      sample_stds_hmc = pd.read_csv(results_path + 'pred_dist/' + 'std_hmc_final3.csv')
      
      #sample_means_hmc = forward_mu(sample_means_hmc, emp_mu, emp_std)
      #sample_stds_hmc = forward_std(sample_stds_hmc, emp_std)
      plt.plot(t_test, sample_means_hmc.T, 'k', alpha=0.2)
      plt.plot(t_test, mu_hmc,'b')
      plt.plot(t_test, mu_test, 'r')
      plt.plot(t_test, y_test, 'ko')

      mu_hmc = pa.get_posterior_predictive_mean(sample_means_hmc)
      se_rmse_hmc = pa.se_of_rmse(y_test, mu_hmc)
      lower_hmc, upper_hmc = pa.get_posterior_predictive_uncertainty_intervals(sample_means_hmc, sample_stds_hmc)
      
      rmse_hmc = pa.rmse(mu_hmc, y_test)
      lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
      
     
      # MF
      
      pa.write_posterior_predictive_samples(trace_mf, 10, t_test, results_path + 'pred_dist/', method='mf_final', gp=gp) 
      
      sample_means_mf = pd.read_csv(results_path + 'pred_dist/' + 'means_mf_final.csv')
      sample_stds_mf = pd.read_csv(results_path + 'pred_dist/' + 'std_mf_final.csv')
            
      mu_mf = pa.get_posterior_predictive_mean(sample_means_mf)
      lower_mf, upper_mf = pa.get_posterior_predictive_uncertainty_intervals(sample_means_mf, sample_stds_mf)
      
      rmse_mf = pa.rmse(mu_mf, y_test)
      se_rmse_mf = pa.se_of_rmse(y_test, mu_mf)
      lppd_mf, lpd_mf = pa.log_predictive_mixture_density(y_test, sample_means_mf, sample_stds_mf)


      # FR
      
      pa.write_posterior_predictive_samples(trace_fr, 10, t_test, results_path +  'pred_dist/', method='fr_final', gp=gp) 
      
      sample_means_fr = pd.read_csv(results_path + 'pred_dist/' + 'means_fr_final.csv')
      sample_stds_fr = pd.read_csv(results_path + 'pred_dist/' + 'std_fr_final.csv')
            
      mu_fr = pa.get_posterior_predictive_mean(sample_means_fr)
      lower_fr, upper_fr = pa.get_posterior_predictive_uncertainty_intervals(sample_means_fr, sample_stds_fr)
      
      rmse_fr = pa.rmse(mu_fr, y_test)
      lppd_fr, lpd_fr = pa.log_predictive_mixture_density(y_test, sample_means_fr, sample_stds_fr)
                  
      
      # Plot HMC vs ML vs MF vs FR / All-in 

      plt.figure(figsize=(18,10))
      plt.subplot(241)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      #plt.plot(df['Year'][0:sep_idx], mu_fit, alpha=0.5, label='train', color='k')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, label='test', color='r')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'NLPD: ' + str(-lpd_), fontsize='small')
      
      plt.subplot(242)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.5, label='test', color='b')
      #plt.plot(df['Year'][sep_idx:], sample_means_hmc.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('HMC' + '\n' + 'RMSE: ' + str(rmse_hmc) + '\n' + 'NLPD: ' + str(-lpd_hmc), fontsize='small')
      
                  
      plt.subplot(243)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, label='test', color='coral')
      #plt.plot(df['Year'][sep_idx:], sample_means_hmc.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('MF' + '\n' + 'RMSE: ' + str(rmse_mf) + '\n' + 'NLPD: ' + str(-lpd_mf), fontsize='small')
      
      plt.subplot(244)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.5, label='test', color='g')
      #plt.plot(df['Year'][sep_idx:], sample_means_fr.T, alpha=0.1, color='gray')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.2)
      plt.legend(fontsize='small')
      plt.title('FR' + '\n' + 'RMSE: ' + str(rmse_fr) + '\n' + 'NLPD: ' + str(-lpd_fr), fontsize='small')
      
      
      # All in 3-way
      
      plt.subplot(245)
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.4, label='HMC')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2, label='ML-II')
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, color='r')
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.5, color='b')
      plt.legend(fontsize='small')
      plt.title('ML-II vs HMC', fontsize='small')

      
      plt.subplot(246)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_hmc, upper_hmc, color='blue', alpha=0.4, label='HMC')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.4, label='FR')
      plt.plot(df['Year'][sep_idx:], mu_hmc, alpha=0.8, color='b')
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.8, color='g')
      plt.legend(fontsize='small')
      plt.title('HMC vs FR',fontsize='small')
      
      
      plt.subplot(247)
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.4, label='MF')
      plt.fill_between(df['Year'][sep_idx:], lower_fr, upper_fr, color='g', alpha=0.4, label='FR')
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, color='coral')
      plt.plot(df['Year'][sep_idx:], mu_fr, alpha=0.5, color='g')
      plt.legend(fontsize='small')
      plt.title('MF vs FR',fontsize='small')
      
      
      plt.subplot(248)      
      plt.plot(df['Year'][sep_idx:], df['Passengers'][sep_idx:], 'ko', markersize=1)   
      plt.fill_between(df['Year'][sep_idx:], lower_mf, upper_mf, color='coral', alpha=0.4, label='MF')
      plt.fill_between(df['Year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='r', alpha=0.2, label='ML-II')
      plt.plot(df['Year'][sep_idx:], mu_mf, alpha=0.5, color='coral')
      plt.plot(df['Year'][sep_idx:], mu_test, alpha=0.5, color='r')
      plt.legend(fontsize='small')
      plt.title('ML-II vs MF',fontsize='small')
      
      
# Metrics

rmse_hmc = pa.rmse(mu_hmc, y_test)
rmse_mf = pa.rmse(mu_mf, y_test)
rmse_fr = pa.rmse(mu_fr, y_test)

lppd_hmc, lpd_hmc = pa.log_predictive_mixture_density(y_test, sample_means_hmc, sample_stds_hmc)
lppd_mf, lpd_mf = pa.log_predictive_mixture_density(y_test, sample_means_mf, sample_stds_mf)
lppd_fr, lpd_fr = pa.log_predictive_mixture_density(y_test, sample_means_fr, sample_stds_fr)

se_rmse_hmc = pa.se_of_rmse(mu_hmc, y_test)
se_rmse_mf = pa.se_of_rmse(mu_mf, y_test)
se_rmse_fr = pa.se_of_rmse(mu_fr, y_test)

se_lpd_hmc = np.std(lppd_hmc)/np.sqrt(len(y_test))
se_lpd_mf = np.std(lppd_mf)/np.sqrt(len(y_test))
se_lpd_fr = np.std(lppd_fr)/np.sqrt(len(y_test))
      
# Persist means
      
np.savetxt(fname=results_path + 'pred_dist/' + 'mu_ml.csv', X=mu_test, delimiter=',', header='')   
np.savetxt(fname=results_path + 'pred_dist/' + 'mu_hmc.csv', X=mu_hmc, delimiter=',', header='')   
np.savetxt(fname=results_path + 'pred_dist/' + 'mu_mf.csv', X=mu_mf, delimiter=',', header='')   
np.savetxt(fname=results_path + 'pred_dist/' + 'mu_fr.csv', X=mu_fr, delimiter=',', header='')   