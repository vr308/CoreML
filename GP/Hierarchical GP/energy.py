#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:42:33 2019

@author: vidhi
"""

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pylab as plt
import  scipy.stats as st 
import seaborn as sns
from sklearn.preprocessing import normalize
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa
import advi_analysis as ad

if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Energy/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Energy/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Energy/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = uni_path
      
      raw = pd.read_csv(path + 'energy.csv', keep_default_na=False)
      
      df = normalize(raw)

      
      y = df[:,-1]
      X = df[:,0:8]
      
      sep_idx = 350
      
      y_train = y[0:sep_idx]
      y_test = y[sep_idx:]
      
      X_train = X[0:sep_idx]
      X_test = X[sep_idx:]
      
      # ML-II 
      
      # sklearn kernel 
      
      # se-ard + noise
      
      se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*8), length_scale_bounds=(0.000001,1e5))
     
      noise = WhiteKernel(noise_level=1**2,
                        noise_level_bounds=(1e-2, 100))  # noise terms
      
      sk_kernel = se_ard + noise
      
      models = []
      
      for i in [0,1,2,3,4,5]:
      
            gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=20)
            models.append(gpr.fit(X_train, y_train))
       
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
      gpr = models[3]
      
      mu_fit, std_fit = gpr.predict(X_train, return_std=True)      
      mu_test, std_test = gpr.predict(X_test, return_std=True)
            
      rmse_ = pa.rmse(mu_test, y_test)
      
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
      
      # No plotting 
      
      s = np.sqrt(gpr.kernel_.k1.k1.constant_value)
      ls =  gpr.kernel_.k1.k2.length_scale
      n = np.sqrt(gpr.kernel_.k2.noise_level)
      
      ml_deltas = {'s':s,'ls':ls, 'n': n}
      varnames = ['s', 'ls','n']
      
      ml_deltas_unravel = {'s':s,
                           'ls__0':ls[0],
                           'ls__1':ls[1],
                           'ls__2':ls[2],
                           'ls__3':ls[3],
                           'ls__4':ls[4],
                           'ls__5':ls[5],
                           'ls__6':ls[6],
                           'ls__7':ls[7],
                           'n': n}
      
      ml_deltas_log = {'log_s': np.log(s), 
                       'log_n': np.log(n), 
                       'log_ls__0': np.log(ls[0]),
                       'log_ls__1': np.log(ls[1]),
                       'log_ls__2': np.log(ls[2]),
                       'log_ls__3': np.log(ls[3]),
                       'log_ls__4': np.log(ls[4]),
                       'log_ls__5': np.log(ls[5]),
                       'log_ls__6': np.log(ls[6]),
                       'log_ls__7': np.log(ls[7])
                       }
      
      