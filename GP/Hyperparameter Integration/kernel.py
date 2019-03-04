#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:23:00 2019

@author: vidhi
"""

class Kernel:
      
      def __init__(self, kind, ):
            
            self.kind = kind 
            self.dimension = dimension 
            self.name = name
            self.hyperparams = hyperparams
      
      def kernel_function():
      
            
            if kernel_type == 'SE':
              
              sig_var = hyper_params[0]
              lengthscale = hyper_params[1]
              
              return pm.gp.cov.Constant(sig_var)*pm.gp.cov.ExpQuad(1, lengthscale)
      
          elif kernel_type == 'PER':
              
              period = hyper_params[0]
              lengthscale = hyper_params[1]
              
              return pm.gp.cov.Periodic(1,period, ls=lengthscale)
              
          elif kernel_type == 'MATERN':
              
              lengthscale = hyper_params[0]
              
              return pm.gp.Matern32(1,ls=lengthscale)
                  
          elif kernel_type == 'RQ':
              
              alpha = hyper_params[0]
              lengthscale = hyper_params[0]
              
              return pm.gp.cov.RatQuad(1, alpha, ls=lengthscale)
      
      def kernel_matrix