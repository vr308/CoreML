#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:41:53 2019

@author: vidhi
"""

import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import scipy.stats as st
import theano.tensor as tt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd

from kernel import Kernel
from data_scheme import GP_Synthetic_Data

# TODO: Generalize to other kernels

class Trace_Hyp_GP:
      
      def __init__(self, trace, gp):
            
            self.trace = trace
            self.gp = gp 
            self.trace_df = self.get_combined_trace()
            self.varnames = list(gp.kernel.hyperparams.keys())
            self.post_trace_mean, self.post_trace_cov, self.list_means, self.list_cov = self.get_post_mcmc_mean_cov()
            
      def get_combined_trace(self):
      
          trace_df = pd.DataFrame()
          for i in list(gp.kernel.hyperparams.keys()):
                trace_df[i] = np.mean(self.trace.get_values(i, combine=False), axis=0)
          return trace_df

      def get_trace_means(self):
            
            trace_means = []
            for i in self.varnames:
                  trace_means.append(self.trace[i].mean())
            return trace_means
      
      @staticmethod
      def get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv):
            
            return  K_s.T.dot(K_inv).dot(y)
      
      @staticmethod
      def get_post_cov_theta(theta, K_s, K_ss, K_inv, pred_noise):
            
            if pred_noise:
                  return K_ss - K_s.T.dot(K_inv).dot(K_s) + theta[-1]*tt.eye(len(self.gp.data.X_star))
            else:
                  return K_ss - K_s.T.dot(K_inv).dot(K_s)
      
      def get_joint_value_from_trace(self, i):
            
            joint = []
            for v in self.varnames:
                  joint.append(trace[v][i])
            return joint
      
      def get_post_mcmc_mean_cov(self):
            
            list_means = []
            list_cov = []
            list_mean_sq = []
            for i in range(0,len(trace_df), 5):
                  print(i)
                  theta = self.get_joint_value_from_trace(trace_df, varnames, i) 
                  kernel = Kernel(self.gp.kernel.name, 1,dict(zip(varnames,theta)) , noise_var=theta[-1])
                  cov = kernel.kernel_func
                  K, K_s, K_ss, K_noise, K_inv = kernel.get_kernel_matrix_blocks(self.gp.data.X, self.gp.data.X_star)
                  mean_i = Trace_Hyp_GP.get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv)
                  cov_i = Trace_Hyp_GP.get_post_cov_theta(theta, K_s, K_ss, K_inv, False)
                  list_means.append(mean_i)
                  list_cov.append(cov_i)
                  list_mean_sq.append(tt.outer(mean_i, mean_i))
            post_mean_trace = tt.mean(list_means, axis=0)
            post_mean_trace = post_mean_trace.eval()
            print('Mean evaluated')
            post_cov_mean =  tt.mean(list_cov, axis=0) 
            post_cov_mean = post_cov_mean.eval() 
            print('Cov evaluated')
            outer_means = tt.mean(list_mean_sq, axis=0)
            outer_means = outer_means.eval()
            print('SSQ evaluated')
            post_cov_trace = post_cov_mean + outer_means - np.outer(post_mean_trace, post_mean_trace)
            return post_mean_trace, post_cov_trace, list_means, list_cov