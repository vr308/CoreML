#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:43:17 2019

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
from sympy import symbols, diff, exp, log, power, sin

varnames = ['s_1', 'ls_2','s_3', 'ls_4','ls_5','s_6','ls_7','alpha_8','s_9','ls_10','n_11'] 

# Read in fr_param 

fr_df_raw = pd.read_csv(results_path + 'VI/fr_df_raw.csv', sep=',', index_col=0)
mu_theta = fr_df_raw['mu_implicit'][varnames]

# Analytical variational inference for CO_2 data

s_1, ls_2, s_3, ls_4, ls_5, s_6, ls_7, alpha_8, s_9, ls_10, n_11, x1, x2 = symbols('s_1 ls_2 s_3 ls_4 ls_5 s_6 ls_7 alpha_8 s_9 ls_10 n_11 x1 x2', real=True)

def kernel(s_1, ls_2, s_3, ls_4, ls_5, s_6, ls_7, alpha_8, s_9, ls_10, n_11, x1, x2):
        
     # se +  sexper + rq + se + noise
      
      sk1 = s_1**2*exp(-0.5*(1/ls_2**2)*(x1 - x2)**2)
      sk2 = s_3**2*exp(-0.5*(1/ls_4**2)*(x1 - x2)**2)*exp((-2*sin(pi*(x1 - x2))**2)*(1/ls_5**2))
      sk3 = s_6**2*exp(1 + (1/(2*alpha_8*ls_7**2))*(x1 - x2)**2)**(-alpha_8)
      sk4 = s_9**2*exp(-0.5*(1/ls_10**2)*(x1 - x2)**2) + n_11**2
      
      return sk1 + sk2 + sk3 + sk4

mu_theta_sub = {s_1: mu_theta['s_1'],
                ls_2:  , 
                s_3: , 
                ls_4:  ,
                ls_5: ,
                s_6: ,
                ls_7:  , 
                alpha_8: , 
                ls_9: , 
                }


def get_symbolic_diff(order, var, kernel, mu_theta_sub):
      
      if len(var) == 1:
            return diff(kernel(s_1, ls_2, s_3, ls_4, ls_5, s_6, ls_7, alpha_8, s_9, ls_10, n_11, x1, x2), var, order).subs(mu_theta_sub)
      else:
            return diff(kernel(s_1, ls_2, s_3, ls_4, ls_5, s_6, ls_7, alpha_8, s_9, ls_10, n_11, x1, x2), var[0], var[1]).subs(mu_theta_sub)
      
   dk_dsf = diff(kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_d2sf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2ls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2sn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_dsfdls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dsfdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dlsdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})

def gradient_K(X):
      
      # Rank 3 tensor -> 3 x n x n 
      
      n_train = len(X) 
      
      dK_dsf_m = np.zeros(shape=(n_train, n_train))
      dK_dls_m = np.zeros(shape=(n_train, n_train))
      dK_dsn_m = np.zeros(shape=(n_train, n_train))
      
      i, j = np.meshgrid(np.arange(n_train), np.arange(n_train))
      index = np.vstack([j.ravel(), i.ravel()]).T
      
      for h in index:
            dK_dsf_m[h[0]][h[1]] = dk_dsf.subs({x1: X[h[0]], x2: X[h[1]]})
            dK_dls_m[h[0]][h[1]] = dk_dls.subs({x1: X[h[0]], x2: X[h[1]]})
            if h[0] == h[1]:
                  dK_dsn_m[h[0]][h[1]] = dk_dsn.subs({x1: X[h[0]], x2: X[h[1]]})
            else:
                  dK_dsn_m[h[0]][h[1]] = 0 
      
      return np.array([dK_dsf_m, dK_dls_m, dK_dsn_m]) 

def curvature_K(X):
      
      # Rank 4 tensor  -> 3 x 3 x n x n 
      
      n_train = len(X) 
      
      d2K_d2sf_m = np.zeros(shape=(n_train, n_train))
      d2K_d2ls_m = np.zeros(shape=(n_train, n_train))
      d2K_d2sn_m = np.zeros(shape=(n_train, n_train))
      d2K_dls_dsf_m = np.zeros(shape=(n_train, n_train))
      d2K_dls_dsn_m = np.zeros(shape=(n_train, n_train))
      d2K_dsn_dsf_m = np.zeros(shape=(n_train, n_train))
      
      i, j = np.meshgrid(np.arange(n_train), np.arange(n_train))
      index = np.vstack([j.ravel(), i.ravel()]).T
      
      for h in index:
          
            d2K_d2sf_m[h[0]][h[1]] = d2k_d2sf.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_d2ls_m[h[0]][h[1]] = d2k_d2ls.subs({x1: X[h[0]], x2: X[h[1]]})
            if h[0] == h[1]:
                  d2K_d2sn_m[h[0]][h[1]] = d2k_d2sn.subs({x1: X[h[0]], x2: X[h[1]]})
            else:
                  d2K_d2sn_m[h[0]][h[1]] = 0
            
            d2K_dls_dsf_m[h[0]][h[1]] = d2k_dsfdls.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_dls_dsn_m[h[0]][h[1]] = d2k_dlsdsn.subs({x1: X[h[0]], x2: X[h[1]]})
            d2K_dsn_dsf_m[h[0]][h[1]] = d2k_dsfdsn.subs({x1: X[h[0]], x2: X[h[1]]})

      T1 = [d2K_d2sf_m, d2K_dls_dsf_m, d2K_dsn_dsf_m]  
      T2 = [d2K_dls_dsf_m, d2K_d2ls_m, d2K_dls_dsn_m]
      T3 = [d2K_dsn_dsf_m, d2K_dls_dsn_m, d2K_d2sn_m]
      
      return np.array([T1, T2, T3])


def gradient_K_star(X, x_star):
      
      n_train = len(X)
      
      row1 = np.zeros(shape=(n_train))
      row2 = np.zeros(shape=(n_train))
      row3 = np.zeros(shape=(n_train))
      
      for i in np.arange(n_train):
            row1[i] = dk_dsf.subs({x1: X[i], x2: x_star})
            row2[i] = dk_dls.subs({x1: X[i], x2: x_star})
            row3[i] = 0
            
      return np.array([row1, row2, row3]).T

def curvature_K_star(X, x_star):
      
      # Rank 3 tensor  -> 3 x 3 x n 
      
     n_train = len(X)
      
     row_d2sf = np.zeros(shape=(n_train))
     row_d2ls = np.zeros(shape=(n_train))
     row_d2sn = np.zeros(shape=(n_train))
     
     row_dsfdls = np.zeros(shape=(n_train))
     row_dlsdsn = np.zeros(shape=(n_train))
     row_dsfdsn = np.zeros(shape=(n_train))
     
     for h in np.arange(len(X)):
           
           row_d2sf[h] = d2k_d2sf.subs({x1: X[h], x2: x_star})
           row_d2ls[h] = d2k_d2ls.subs({x1: X[h], x2: x_star})
           #row_d2sn[h] = d2k_d2sn.subs({x1: X[h], x2: x_star})
           row_d2sn[h] = 0
           
           row_dsfdls[h] = d2k_dsfdls.subs({x1: X[h], x2: x_star})
           row_dlsdsn[h] = d2k_dlsdsn.subs({x1: X[h], x2: x_star})
           row_dsfdsn[h] = d2k_dsfdsn.subs({x1: X[h], x2: x_star})
           
     M1 = np.array([row_d2sf, row_dsfdls, row_dsfdsn])
     M2 = np.array([row_dsfdls, row_d2ls, row_dlsdsn])
     M3 = np.array([row_dsfdsn, row_dlsdsn, row_d2sn])
      
     return np.array([M1.T, M2.T, M3.T])

def gradient_K_star_star(x_star):
      
      return np.array([dk_dsf.subs({x1: x_star, x2: x_star}), dk_dls.subs({x1: x_star, x2: x_star}), 0])

def curvature_K_star_star(x_star):
            
       # 3 X 3 matrix
      row1 = [d2k_d2sf.subs({x1: x_star, x2: x_star}), d2k_dsfdls.subs({x1: x_star, x2: x_star}),                 d2k_dsfdsn.subs({x1: x_star, x2: x_star})]
      row2 = [d2k_dsfdls.subs({x1: x_star, x2: x_star}), d2k_d2ls.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star})]
      #row3 = [d2k_dsfdsn.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star}), d2k_d2sn.subs({x1: x_star, x2: x_star})]
      row3 = [0, 0, 0]
      
      return np.array([row1, row2, row3], dtype=np.float)

def gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s):
      
      dK_starT = gradient_K_star(X, x_star).T
      
      return np.matmul(np.matmul(dK_starT, K_inv), y)[:, np.newaxis] + np.matmul(np.matmul(K_s.T, dK_inv), y)

def curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv):
      
      dK_starT = gradient_K_star(X, x_star).T
      d2K_star = curvature_K_star(X, x_star)
      d2K_starT = np.array([d2K_star[0].T, d2K_star[1].T, d2K_star[2].T])
      
      return  np.matmul(np.matmul(d2K_starT, K_inv), y) + 2*np.matmul(np.matmul(dK_starT, dK_inv), y) + np.matmul(np.matmul(K_s.T, d2K_inv),y).reshape(3,3)

def curvature_gp_pred_var(X, x_star, y, K_s,  K_inv, dK_inv, d2K_inv):
      
      dK_starT = gradient_K_star(X, x_star).T
      dK_star = dK_starT.T
      d2K_star_star = curvature_K_star_star(x_star)
      d2K_star = curvature_K_star(X, x_star)
      d2K_starT = np.array([d2K_star[0].T, d2K_star[1].T, d2K_star[2].T])
      
      return  d2K_star_star - 2*np.matmul(np.matmul(d2K_starT, K_inv), K_s).reshape(3,3) - 4*np.matmul(np.matmul(dK_starT, dK_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(dK_starT, K_inv), dK_star) -  np.matmul(np.matmul(K_s.T, d2K_inv), K_s).reshape(3,3) 


#d2K_star_star - np.matmul(np.matmul(d2K_starT, K_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(dK_starT,dK_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(dK_starT, K_inv), dK_starT.T) -  np.matmul(np.matmul(K_s.T, d2K_inv), K_s).reshape(3,3) - 2*np.matmul(np.matmul(K_s.T, dK_inv), dK_starT.T).reshape(3,3) - np.matmul(np.matmul(K_s.T,K_inv), d2K_starTT).reshape(3,3)
      
#a1 = np.matmul(np.matmul(dK_starT, dK_inv), K_s).reshape(3,3)
#a2 = np.matmul(np.matmul(K_s.T, dK_inv), dK_star).reshape(3,3)

def deterministic_gp_pred_mean(X, x_star, y, K_s, K_ss, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta, pred_vi_mean):
      
      #K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      
      d2_gp_mean = curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv) # 3x3 matrix
      
      return pred_vi_mean + 0.5*np.trace(np.matmul(d2_gp_mean, cov_theta))

def deterministic_gp_pred_covariance(X, x_star, y, K_s, K_ss, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta, pred_vi_var):
      
      #K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      
      d1_gp_mean = gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s)
      d2_gp_var = curvature_gp_pred_var(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv)
      
      return pred_vi_var + 0.5*np.trace(np.matmul(d2_gp_var, cov_theta)) + np.trace(np.matmul(np.matmul(d1_gp_mean, d1_gp_mean.T), cov_theta))


