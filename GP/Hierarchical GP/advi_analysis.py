#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:57:22 2019

@author: vidhi

Analysis for ADVI (MF and FR) runs of hierarchical GP 

"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.stats as st

# Constructing posterior predictive distribution

def transform_tracker_values(tracker, param_dict, name_mapping, raw_mapping):

      mean_df = pd.DataFrame(np.array(tracker['mean']), columns=list(param_dict['mu'].keys()))
      sd_df = pd.DataFrame(np.array(tracker['std']), columns=list(param_dict['mu'].keys()))
      for i in mean_df.columns:
            print(i)
            if (i[-2:] == '__'):
                 mean_df[name_mapping[i]] = np.exp(raw_mapping.get(i).distribution.transform_used.backward(mean_df[i]).eval()) 
            else:
                mean_df[name_mapping[i]] = np.exp(mean_df[i]) 
      return mean_df, sd_df


def convergence_report(tracker, varnames, title):
      
      # Plot Negative ElBO track with params in true space
      
      #mean_mf_df, sd_mf_df = transform_tracker_values(tracker_mf, mf_param)
#      mean_fr_df, sd_fr_df = transform_tracker_values(tracker, param_dict)

#      fig = plt.figure(figsize=(16, 9))
#      for i in np.arange(3):
#            print(i)
#            if (np.mod(i,8) == 0):
#                   fig = plt.figure(figsize=(16,9))
#                   i = i + 8
#                   print(i)
#            plt.subplot(2,4,np.mod(i, 8)+1)
            #plt.plot(mean_mf_df[varnames[i+8]], color='coral')
#            plt.plot(mean_fr_df[varnames[i+8]], color='green')
#            plt.title(varnames[i+8])
#            plt.axhline(param_dict['mu_implicit'][varnames[i+8]], color='r')
      
      
      fig = plt.figure(figsize=(16, 9))
      mu_ax = fig.add_subplot(221)
      std_ax = fig.add_subplot(222)
      hist_ax = fig.add_subplot(212)
      mu_ax.plot(tracker['mean'])
      mu_ax.set_title('Mean track')
      std_ax.plot(tracker['std'], label=varnames)
      std_ax.set_title('Std track')
      plt.legend(fontsize='x-small')
      hist_ax.plot(tracker.hist)
      hist_ax.set_title('Negative ELBO track');
      fig.suptitle(title)


# Implicit variational posterior density
            
def get_implicit_variational_posterior(var, means, std, x):
      
      sigmoid = lambda x : 1 / (1 + np.exp(-x))
      
      if (var.name[-2:] == '__'):
            # Then it is an interval variable
            
            eps = lambda x : var.distribution.transform_used.forward_val(np.log(x))
            #backward_theta = lambda x: var.distribution.transform_used.backward(x).eval()   
            width = (var.distribution.transform_used.b -  var.distribution.transform_used.a).eval()
            total_jacobian = lambda x: x*(width)*sigmoid(eps(x))*(1-sigmoid(eps(x)))
            pdf = lambda x: st.norm.pdf(eps(x), means[var.name], std[var.name])/total_jacobian(x)
            return pdf(x)
      
      else:
            # Then it is just a log variable
            
            pdf = lambda x: st.norm.pdf(np.log(x), means[var.name], std[var.name])/x   
            return pdf(x)
            

# Converting raw params back to param space

def analytical_variational_opt(model, param_dict, summary_trace, raw_mapping, name_mapping):
      
      keys = list(param_dict['mu'].keys())
      
      # First tackling transformed means
      
      mu_implicit = {}
      rho_implicit = {}
      for i in keys:
            if (i[-2:] == '__'):
                  name = name_mapping[i]
                  #mean_value = np.exp(raw_mapping.get(i).distribution.transform_used.backward(param_dict['mu'][i]).eval())
                  mean_value = summary_trace['mean'][name]
                  sd_value = summary_trace['sd'][name]
                  mu_implicit.update({name : np.array(mean_value)})
                  rho_implicit.update({name : np.array(sd_value)})
            else:
                  name = name_mapping[i]
                  #mean_value = np.exp(param_dict['mu'][i])
                  mean_value = summary_trace['mean'][name]
                  sd_value = summary_trace['sd'][name]
                  name = name_mapping[i]
                  mu_implicit.update({name : np.array(mean_value)})
                  rho_implicit.update({name : np.array(sd_value)})
      param_dict.update({'mu_implicit' : mu_implicit})
      param_dict.update({'rho_implicit' : rho_implicit})

      return param_dict


# Helper for getting analytical vi mean and cov

def get_vi_analytical(X, y, X_star, dh, d2h, d2g, theta, mu_theta, cov_theta):
                  
    #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), theta)      
    #pred_vi_mean =  np.matmul(np.matmul(K_s.T, K_inv), y)
    #pred_vi_var =  np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))
    
    pred_g_mean = gp_mean(theta, X, y, X_star)
    pred_g_cov = np.diag(gp_cov(theta, X, y, X_star))

    pred_ng_mean = []
    pred_ng_var = []
    
    # To fix this 
    
    pred_ng_mean = pred_g_mean + 0.5*np.trace(np.matmul(d2h(theta, X, y, X_star), np.array(cov_theta)))
    pred_ng_var = pred_vi_var + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta))

    for i in np.arange(len(X_star)): # To vectorize this loop
          
          print(i)
          x_star = X_star[i].reshape(1,1)

          pred_ng_mean.append(pred_g_mean[i] + 0.5*np.trace(np.matmul(d2h(theta, X, y, x_star), np.array(cov_theta))))
          print(pred_ng_mean[i])
          pred_ng_var.append(pred_vi_var[i] + 0.5*np.trace(np.matmul(d2g(theta, X, y, x_star), cov_theta)) + np.trace(np.matmul(np.outer(dh(theta, X, y, x_star),dh(theta, X, y, x_star).T), cov_theta)))

    return pred_ng_mean, pred_ng_var