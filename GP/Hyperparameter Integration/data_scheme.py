#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:23:18 2019

@author: vidhi

Generate synthetic datasets for a testing GPs, the true function is simulated as a draw from a GP with a specified covariance function

"""

class GP_Synthetic_Data:
      
        def __init__(self, kernel, xmin, xmax, n_all, n_train):
              
               self.X_all = np.linspace(xmin, xmax, n_all)[:,None] # The granularity of the grid
               self.

        
        @staticmethod
        def generate_gp_latent(X_star, mean, cov):
    
              return np.random.multivariate_normal(mean(X_star).eval(), cov=cov(X_star, X_star).eval())

        @staticmethod
        def generate_gp_training(X_all, f_all, n_train, noise_var, uniform):
          
                if uniform == True:
                     X = np.random.choice(X_all.ravel(), n_train, replace=False)
                else:
                     pdf = 0.5*st.norm.pdf(X_all, 2, 0.7) + 0.5*st.norm.pdf(X_all, 7.5,1)
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
                return X, y, X_star, f_star, f, train_index
            
      