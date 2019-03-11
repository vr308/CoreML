#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:23:00 2019

@author: vidhi

A class that supports a range of tasks with base kernel functions and composite kernel functions
in user-defined dimensions.

"""

import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse, det

# TODO: Set up plumbing for high dimensional and ARD kernels
# TODO: Non-stationary kernels

class Kernel:
      
      def __init__(self, name, dimension, hyperparams, kernel, noise_var):
            
            self.name = name # string
            self.dimension = dimension   # int 
            self.hyperparams = hyperparams # dict
            self.kernel = None # instance of gp.cov 
            self.noise_var = noise_var # float
      
      def set_base_kernel_function(self):
      
          if self.name == 'SE':
              
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              self.kernel = pm.gp.cov.Constant(sig_var)*pm.gp.cov.ExpQuad(self.dimension, ls=lengthscale)
              
              return self.kernel
      
          elif self.name == 'PER':
              
              sig_var = self.hyperparams['sig_var']
              period = self.hyperparams['period']
              lengthscale = self.hyperparams['lengthscale']
              self.kernel = pm.gp.cov.Constant(sig_var)*pm.gp.cov.Periodic(self.dimension,period, ls=lengthscale)
              
              return self.kernel
              
          elif self.name  == 'MATERN32':
              
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              self.kernel = pm.gp.cov.Constant(sig_var)*pm.gp.cov.Matern32(self.dimension, ls=lengthscale)
              
              return self.kernel 
        
          elif self.name  == 'MATERN52':
                
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              self.kernel = pm.gp.cov.Constant(sig_var)*pm.gp.cov.Matern52(self.dimension, ls=lengthscale)
              
              return self.kernel
                  
          elif self.name  == 'RQ':
              
              sig_var = self.hyperparams['sig_var']
              alpha = self.hyperparams['alpha']
              lengthscale = self.hyperparams['lengthscale']
              self.kernel = pm.gp.cov.Constant(sig_var)*pm.gp.cov.RatQuad(self.dimension, alpha, ls=lengthscale)
              
              return self.kernel
      
      def get_kernel_hyp_string(self):
      
          if self.name == 'SE':

              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + r', $\gamma$: ' + str(self.hyperparams['lengthscale']) + r', $\sigma_{n}^{2}$: ' + str(self.noise_var) + '}'
      
          elif self.name == 'PER':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + r'$\{p$: ' + str(self.hyperparams['period']) + r', $\gamma$: ' + str(self.hyperparams['lengthscale']) + '}'
              
          elif self.name == 'MATERN32':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + r'$\gamma$: ' + str(self.hyperparams['lengthscale'])  + '}'
                    
          elif self.name == 'MATERN52':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + r'$\gamma$: ' + str(self.hyperparams['lengthscale'])  + '}'
              
          elif self.name == 'RQ':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + r'$\{\alpha$: ' + str(self.hyperparams['alpha']) + r', $\gamma$: ' + str(self.hyperparams['lengthscale']) + '}'
          
      def get_kernel_matrix_blocks(self, X, X_star):
          
          K = self.kernel(X)
          K_s = self.kernel(X, X_star)
          K_ss = self.kernel(X_star, X_star)
          K_noise = K + self.noise_var*tt.eye(len(X))
          K_inv = matrix_inverse(K_noise)
          return K, K_s, K_ss, K_noise, K_inv
         
      @staticmethod
      def plot_kernel_matrix(K, title):
    
          plt.matshow(K.eval())
          plt.colorbar()
          plt.title(title, fontsize='x-small')
    
      @staticmethod
      def plot_prior_samples(X_star, K_ss):
      
          std = np.sqrt(np.diag(K_ss))
          plt.figure()
          plt.plot(X_star,st.multivariate_normal.rvs(cov=K_ss, size=10).T)
          plt.fill_between(np.ravel(X_star), 0 - 1.96*std, 
                     0 + 1.96*std, alpha=0.2, color='g',
                     label='95% CR')
      

if __name__ == "__main__":

      cov = Kernel('RQ', 1, {'sig_var': 10, 'alpha':2, 'lengthscale':2.0}, kernel = None, noise_var=1)
      cov.set_base_kernel_function()
      hyp_string = cov.get_kernel_hyp_string()
      
      K, K_s, K_ss, K_noise, K_inv = cov.get_kernel_matrix_blocks(X, X_star)
      
      Kernel.plot_prior_samples(X_star, K_ss.eval())
      
      Kernel.plot_kernel_matrix(K,' ')