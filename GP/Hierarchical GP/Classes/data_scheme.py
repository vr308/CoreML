#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:23:18 2019

@author: vidhi

Generate synthetic datasets for a testing GPs, the true function is simulated as a draw from a GP with a specified covariance function.

There are two modes for assigning training data:
      
1) Uniform
2) Non-Uniform

"""
import numpy as np
import scipy.stats as st
import pymc3 as pm
import pandas as pd
import matplotlib.pylab as plt
from kernel import Kernel

class GP_Synthetic_Data:
      
        def __init__(self, xmin, xmax, n_all, n_train, mean, kernel, uniform):
              
               self.mean = mean
               self.kernel = kernel.kernel_func
               self.noise_var = kernel.noise_var
               self.uniform = uniform
               self.n_train = n_train
               self.n_all = n_all
               self.snr = np.round(kernel.hyperparams['sig_var']/kernel.noise_var, 3)
               self.X_all = np.linspace(xmin, xmax, n_all)[:,None] # The granularity of the grid
               self.f_all = GP_Synthetic_Data.generate_gp_latent(self)
               self.X, self.y, self.X_star, self.f_star, self.f, \
                     self.train_index, self.test_index = GP_Synthetic_Data.generate_gp_train_test(self)

        def generate_gp_latent(self):
    
              return np.random.multivariate_normal(self.mean(self.X_all).eval(), cov=self.kernel(self.X_all, self.X_all).eval())

        def generate_gp_train_test(self):
          
                if self.uniform == True:
                     X = np.random.choice(self.X_all.ravel(), self.n_train, replace=False)
                else:
                     pdf = 0.5*st.norm.pdf(self.X_all, 2, 0.7) + 0.5*st.norm.pdf(self.X_all, 7.5,1)
                     prob = pdf/np.sum(pdf)
                     X = np.random.choice(self.X_all.ravel(), self.n_train, replace=False,  p=prob.ravel())
                 
                train_index = []
                for i in X:
                      train_index.append(self.X_all.ravel().tolist().index(i))
                test_index = [i for i in np.arange(len(self.X_all)) if i not in train_index]
                X = self.X_all[train_index]
                f = self.f_all[train_index]
                X_star = self.X_all[test_index]
                f_star = self.f_all[test_index]
                y = f + np.random.normal(0, scale=np.sqrt(self.noise_var), size=self.n_train)
                return X, y, X_star, f_star, f, train_index, test_index
          
        @staticmethod
        def persist_datasets(X, X_star, f, f_star, y, tag):
      
           X.tofile('X' + '_' + tag + '.csv', sep=',')
           X_star.tofile('X_star' + '_' + tag + '.csv', sep=',')
           y.tofile('y' + '_' + tag + '.csv', sep=',')
           f_star.tofile('f_star' + '_' + tag + '.csv', sep=',')

        @staticmethod
        def load_datasets(tag, filepath):
      
            X = np.asarray(pd.read_csv('X.csv', header=None))
            y = np.asarray(pd.read_csv('y.csv', header=None))
            X_star = np.asarray(pd.read_csv('X_star.csv', header=None))
            f_star = np.asarray(pd.read_csv('f_star.csv', header=None))
            return X.reshape(len(X[0]),1), y.reshape(len(y[0]),),  \
            X_star.reshape(len(X_star[0]),1), f_star.reshape(len(f_star[0]),)
        
        def plot_y(self, title):

          plt.figure()
          plt.plot(self.X_star, self.f_star, linestyle='dashed', color='dodgerblue',lw=2, label="True f")
          plt.plot(self.X, self.y, 'ok', ms=3, alpha=0.8, label="Data")
          plt.xlabel("X") 
          plt.ylabel("The true f(x)") 
          plt.legend()
          plt.title(title, fontsize='x-small')
                  
if __name__ == "__main__":
                
          mean = pm.gp.mean.Zero()
          kernel = Kernel('RQ', 1, {'sig_var': 2, 'alpha':0.5, 'lengthscale':1.0}, noise_var=0.4)
          data = GP_Synthetic_Data(0,13,200,40, mean, kernel, uniform=True)
          data.plot_y('Training data (iid noise)')
