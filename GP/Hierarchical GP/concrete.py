#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:40:26 2019

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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv
import posterior_analysis as pa
import advi_analysis as ad

if __name__ == "__main__":
      
      # Load data 
      
      home_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Data/Concrete/'
      
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Concrete/'
      #results_path = '~/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/Airline/'

      path = uni_path
      
      df = pd.read_csv(path + 'concrete.csv', keep_default_na=False)
      
      y = df['strength']
      X = df[['cement','slag','ash','water','superplasticize','coarse_aggregate','fine_aggregate','age']]
      
      sep_idx = 800
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      X_train = X[0:sep_idx].values
      X_test = X[sep_idx:].values
      
      # ML-II 
      
      # sklearn kernel 
      
      # se-ard + noise
      
      se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*8), length_scale_bounds=(0.000001,1e5))
     
      noise = WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-3, 100))  # noise terms
      
      sk_kernel = se_ard + noise
      
      gpr = GaussianProcessRegressor(kernel=sk_kernel, normalize_y=True, n_restarts_optimizer=20)
      gpr.fit(X_train, y_train)
       
      print("\nLearned kernel: %s" % gpr.kernel_)
      print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
      
      print("Predicting with trained gp on training / test")
      
      mu_fit, std_fit = gpr.predict(X_train, return_std=True)      
      mu_test, std_test = gpr.predict(X_test, return_std=True)
      
      rmse_ = pa.rmse(mu_test, y_test)
      
      lpd_ = pa.log_predictive_density(y_test, mu_test, std_test)
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       