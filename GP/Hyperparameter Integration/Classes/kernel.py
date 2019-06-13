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
import scipy.stats as st
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse, det
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel

# TODO: Set up plumbing for high dimensional and ARD kernels
# TODO: Non-stationary kernels
# TODO: Add support for Linear, Polynomial

class Kernel:
      
      def __init__(self, name, dimension, hyperparams, noise_var, **kwargs):
            
            self.name = name # string
            self.dimension = dimension   # int 
            self.hyperparams = hyperparams # dict
            self.noise_var = noise_var # float
            if len(kwargs) > 0:
                   self.kernel_func = kwargs['kernel_func']
                   self.sklearn_kernel = kwargs['sklearn_kernel']
            else:
                  self.kernel_func, self.sklearn_kernel = Kernel.set_base_kernel_function(self) # instance of gp.cov                  
                  
      def set_base_kernel_function(self):
            
          if self.name == 'LIN':
               
               return;
               
          if self.name == 'POLY':
               
               return;
               
          if self.name == 'SE':
              
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              return pm.gp.cov.Constant(sig_var)*pm.gp.cov.ExpQuad(self.dimension, ls=lengthscale), \
              Ck(sig_var, (1e-10, 1e2)) * RBF(lengthscale, length_scale_bounds=(0.5, 10)) + \
              WhiteKernel(self.noise_var, noise_level_bounds=(1e-5,100))
      
          elif self.name == 'PER':
              
              sig_var = self.hyperparams['sig_var']
              period = self.hyperparams['period']
              lengthscale = self.hyperparams['lengthscale']
              return pm.gp.cov.Constant(sig_var)*pm.gp.cov.Periodic(self.dimension,period, ls=lengthscale), \
              Ck(sig_var, (1e-10, 1e2)) * PER(length_scale = lengthscale, periodicity = 1.0, length_scale_bounds=(0.5, 10)) + WhiteKernel(self.noise_var, noise_level_bounds=(1e-5,100))
                            
          elif self.name  == 'MATERN32':
              
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              return  pm.gp.cov.Constant(sig_var)*pm.gp.cov.Matern32(self.dimension, ls=lengthscale), \
              Ck(sig_var, (1e-10, 1e2)) * Matern(length_scale = lengthscale, length_scale_bounds=(0.5, 10), nu=1.5) + \
              WhiteKernel(self.noise_var, noise_level_bounds=(1e-5,100))
                      
          elif self.name  == 'MATERN52':
                
              sig_var = self.hyperparams['sig_var']
              lengthscale = self.hyperparams['lengthscale']
              return  pm.gp.cov.Constant(sig_var)*pm.gp.cov.Matern52(self.dimension, ls=lengthscale), \
              Ck(sig_var, (1e-10, 1e2)) * Matern(length_scale = lengthscale, length_scale_bounds=(0.5, 10), nu=2.5) + WhiteKernel(self.noise_var, noise_level_bounds=(1e-5,100))
               
          elif self.name  == 'RQ':
              
              sig_var = self.hyperparams['sig_var']
              alpha = self.hyperparams['alpha']
              lengthscale = self.hyperparams['lengthscale']
              return  pm.gp.cov.Constant(sig_var)*pm.gp.cov.RatQuad(self.dimension, alpha, ls=lengthscale), \
              Ck(sig_var, (1e-10, 1e2)) * RQ(length_scale = lengthscale, alpha = alpha, length_scale_bounds=(0.5, 10)) + WhiteKernel(self.noise_var, noise_level_bounds=(1e-5,100))
              
                      
      def get_kernel_hyp_string(self):
      
          if self.name == 'SE':

              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + \
              r', $\gamma$: ' + str(self.hyperparams['lengthscale']) + \
              r', $\sigma_{n}^{2}$: ' + str(self.noise_var) + '}'
      
          elif self.name == 'PER':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + \
              r'$\{p$: ' + str(self.hyperparams['period']) + r', $\gamma$: ' + \
              str(self.hyperparams['lengthscale']) + '}'
              
          elif self.name == 'MATERN32':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + \
              r'$\gamma$: ' + str(self.hyperparams['lengthscale'])  + '}'
                    
          elif self.name == 'MATERN52':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + \
              r'$\gamma$: ' + str(self.hyperparams['lengthscale'])  + '}'
              
          elif self.name == 'RQ':
              
              return r'$\{\sigma_{f}^{2}$: ' + str(self.hyperparams['sig_var']) + \
              r'$\{\alpha$: ' + str(self.hyperparams['alpha']) + \
              r', $\gamma$: ' + str(self.hyperparams['lengthscale']) + '}'
              
      def get_kernel_matrix_blocks(self, X, X_star):
          
          K = self.kernel_func(X)
          K_s = self.kernel_func(X, X_star)
          K_ss = self.kernel_func(X_star, X_star)
          K_noise = K + self.noise_var*tt.eye(len(X))
          K_inv = matrix_inverse(K_noise)
          return K, K_s, K_ss, K_noise, K_inv
    
      @staticmethod
      def get_composite_hyp_dict(kernel1, kernel2):
            
            keys = [x + '_1' for x in list(kernel1.hyperparams.keys())] + [x + '_2' for x in list(kernel2.hyperparams.keys())]
            values = list(kernel1.hyperparams.values()) + list(kernel2.hyperparams.values())
      
            return dict(zip(keys, values))
    
      @staticmethod
      def product_(kernel1, kernel2):
            
         name = kernel1.name + 'x' + kernel2.name
         dimension = kernel1.dimension 
         hyperparams = Kernel.get_composite_hyp_dict(kernel1, kernel2)
         k_func = kernel1.kernel_func*kernel2.kernel_func
         sklearn_func = kernel1.sklearn_kernel*kernel2.sklearn_kernel
         noise_var = kernel1.noise_var # Independent noise component
         
         return Kernel(name, dimension, hyperparams, noise_var, kernel_func=k_func, sklearn_kernel=sklearn_func)
         
   
      @staticmethod
      def sum_(kernel1, kernel2):
            
         name = kernel1.name + '+' + kernel2.name
         dimension = kernel1.dimension 
         hyperparams = Kernel.get_composite_hyp_dict(kernel1, kernel2)
         k_func = kernel1.kernel_func + kernel2.kernel_func
         sklearn_func = kernel1.sklearn_kernel + kernel2.sklearn_kernel
         noise_var = kernel1.noise_var # Independent noise component
         
         return Kernel(name, dimension, hyperparams, noise_var, kernel_func=k_func, sklearn_kernel=sklearn_func)
         
   
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

      #cov = Kernel('RQ', 1, {'sig_var': 10, 'alpha':2, 'lengthscale':2.0}, noise_var=1)
#      hyp_string = cov.get_kernel_hyp_string()
#      
#      K, K_s, K_ss, K_noise, K_inv = cov.get_kernel_matrix_blocks(X, X_star)
#      
#      Kernel.plot_prior_samples(X_star, K_ss.eval())
#      
#      Kernel.plot_kernel_matrix(K,' ')
      
      kernel1 = Kernel('PER', 1, {'sig_var': 5, 'lengthscale':2, 'period' : 3}, noise_var=1)
      kernel2 = Kernel('SE', 1, {'sig_var':4, 'lengthscale':10}, noise_var=1)
      
      prod_kernel = Kernel.product_(kernel1, kernel2)
      sum_kernel = Kernel.sum_(kernel1, kernel2)
      
      K, K_s, K_ss, K_noise, K_inv = sum_kernel.get_kernel_matrix_blocks(data.X, data.X_star)
      
      Kernel.plot_prior_samples(data.X_star, K_ss.eval())
      