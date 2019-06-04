#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:13:32 2019

@author: vidhi
"""

import numpy as np 
import matplotlib.pylab as plt
import theano.tensor as tt
import pymc3 as pm
from theano.tensor.nlinalg import matrix_inverse
import pandas as pd
from sampled import sampled
from scipy.misc import derivative
import csv
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from sympy import symbols, diff, exp, log, power


def se_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      #indicator = 1 if x1 == x2 else 0
      return sig_sd**2*exp(-0.5*(1/ls**2)*(x1 - x2)**2) + noise_sd**2

def gradient_K(X):
      
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
      
      n = len(X)
      row1 = np.zeros(shape=(n))
      row2 = np.zeros(shape=(n))
      row3 = np.zeros(shape=(n))
      
      for i in np.arange(n):
            row1[i] = dk_dsf.subs({x1: X[i], x2: x_star})
            row2[i] = dk_dls.subs({x1: X[i], x2: x_star})
            #row3[i] = dk_dsn.subs({x1: X[i], x2: x_star})
            row3[i] = 0
            
      return np.array([row1, row2, row3]).T

def curvature_K_star(X, x_star):
      
      # Rank 3 tensor  -> 3 x 3 x n 
      
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
      #return np.array([dk_dsf.subs({x1: x_star, x2: x_star})], dtype=np.float)

def curvature_K_star_star(x_star):
            
       # 3 X 3 matrix
      row1 = [d2k_d2sf.subs({x1: x_star, x2: x_star}), d2k_dsfdls.subs({x1: x_star, x2: x_star}),                 d2k_dsfdsn.subs({x1: x_star, x2: x_star})]
      row2 = [d2k_dsfdls.subs({x1: x_star, x2: x_star}), d2k_d2ls.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star})]
      #row3 = [d2k_dsfdsn.subs({x1: x_star, x2: x_star}), d2k_dlsdsn.subs({x1: x_star, x2: x_star}), d2k_d2sn.subs({x1: x_star, x2: x_star})]
      row3 = [0, 0, 0]
      
      return np.array([row1, row2, row3], dtype=np.float)

def gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s):
      
      dKsT = gradient_K_star(X, x_star).T
      
      return np.matmul(np.matmul(dKsT, K_inv), y)[:, np.newaxis] + np.matmul(np.matmul(K_s.T, dK_inv), y)

def curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv):
      
      dKsT = gradient_K_star(X, x_star).T
      d2KsT = curvature_K_star(X, x_star).T
      
      d2KsT = [d2KsT[0].T, d2KsT[1].T, d2KsT[2].T ]
      
      return  np.matmul(np.matmul(d2KsT, K_inv),y) + 2*np.matmul(np.matmul(dKsT, dK_inv), y) + np.matmul(np.matmul(K_s.T, d2K_inv),y).reshape(3,3).T

def curvature_gp_pred_var(X, x_star, y, K_s,  K_inv, dK_inv, d2K_inv):
      
      dK_starT = gradient_K_star(X, x_star).T
      dK_star = dK_starT.T
      d2K_star_star = curvature_K_star_star(x_star)
      d2K_star = curvature_K_star(X, x_star)
      
      d2K_starT = [d2K_star[0].T, d2K_star[1].T, d2K_star[2].T]
      
      #J = curvature_K_star(X, x_star)
      
      #d2K_starTT = np.array([J[:,:,0], J[:,:,1], J[:,:,2]])
      
      return  d2K_star_star - 2*np.matmul(np.matmul(d2K_starT, K_inv), K_s).reshape(3,3).T - 4*np.matmul(np.matmul(dK_starT, dK_inv), K_s).reshape(3,3).T - 2*np.matmul(np.matmul(dK_starT, K_inv), dK_star) -  np.matmul(np.matmul(K_s.T, d2K_inv), K_s).reshape(3,3) 

def deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta):
      
      K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      pred_vi_mean = np.matmul(np.matmul(K_s.T.eval(), K_inv), y).reshape(1,1)
      
      d2_gp_mean = curvature_gp_pred_mean(X, x_star, y, K_s.eval(), K_inv, dK_inv, d2K_inv) # 3x3 matrix
      
      return pred_vi_mean + 0.5*np.trace(np.matmul(d2_gp_mean, cov_theta))

def deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta, cov_theta):
      
      K_s, K_ss = get_star_kernel_matrix_blocks(X, x_star, mu_theta)
      pred_vi_var =  K_ss.eval() - np.matmul(np.matmul(K_s.T.eval(), K_inv), K_s.eval())
      #pred_vi_mean, pred_vi_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
      
      d1_gp_mean = gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s.eval())
      d2_gp_var = curvature_gp_pred_var(X, x_star, y, K_s.eval(), K_inv, dK_inv, d2K_inv)
      
      return pred_vi_var + 0.5*np.trace(np.matmul(d2_gp_var, cov_theta)) + np.trace(np.matmul(np.matmul(d1_gp_mean, d1_gp_mean.T), cov_theta))

def load_datasets(path, n_train):
      
       n_test = 200 - n_train
       X = np.asarray(pd.read_csv(path + 'X_' + str(n_train) + '.csv', header=None)).reshape(n_train,1)
       y = np.asarray(pd.read_csv(path + 'y_' + str(n_train) + '.csv', header=None)).reshape(n_train,)
       X_star = np.asarray(pd.read_csv(path + 'X_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,1)
       f_star = np.asarray(pd.read_csv(path + 'f_star_' + str(n_train) + '.csv', header=None)).reshape(n_test,)
       return X, y, X_star, f_star
 
def get_star_kernel_matrix_blocks(X, X_star, point):
    
          cov = pm.gp.cov.Constant(point['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=point['ls'])
          K_s = cov(X, X_star)
          K_ss = cov(X_star, X_star)
          return K_s, K_ss
    
      
if __name__ == "__main__":

   
      # Loading data
      
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      mac_path = '/Users/vidhi.lalchand/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      desk_home_path = '/home/vr308/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
      
      path = uni_path


      # Edit here to change generative model
      
      input_dist =  'Unif'
      snr = 10
      suffix = input_dist + '/' + 'SNR_' + str(snr) + '/'
      true_hyp = {'sig_sd' : np.round(np.sqrt(100),3), 'ls' : 5, 'noise_sd' : np.round(np.sqrt(10),3)}

      data_path = path + 'Data/1d/' + suffix
      results_path = path + 'Results/1d/' + suffix 
      
      varnames = ['sig_sd', 'ls', 'noise_sd']
      log_varnames = ['log_s', 'log_ls', 'log_n']
    
      X_20, y_20, X_star_20, f_star_20 = load_datasets(data_path, 20)
    
      cov_theta_mf =  np.cov(trace_mf_df[varnames], rowvar=False)
      cov_theta_fr = np.cov(trace_fr_df[varnames], rowvar=False)
      
      mu_theta_mf = mf_param['mu_implicit']
      mu_theta_fr = fr_param['mu_implicit']
      
      sig_sd, ls, noise_sd, x1, x2 = symbols('sig_sd ls noise_sd x1 x2', real=True)
      
      mu_theta = mu_theta_mf

      dk_dsf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      dk_dsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_d2sf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2ls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_d2sn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd, 2).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      d2k_dsfdls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, ls).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dsfdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      d2k_dlsdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, noise_sd).subs({sig_sd:mu_theta['sig_sd'], ls: mu_theta['ls'], noise_sd: mu_theta['noise_sd']})
      
      n_train = 20
            
      cov = pm.gp.cov.Constant(mu_theta['sig_sd']**2)*pm.gp.cov.ExpQuad(1, ls=mu_theta['ls'])
      
      K = cov(X).eval()
      K_noise = (K + np.square(mu_theta['noise_sd'])*tt.eye(n_train))
      K_inv = matrix_inverse(K_noise).eval()
         
      dK = gradient_K(X) # 3 x n x n (rank 3 tensor)
      d2K = curvature_K(X) # 3 x 3 x n x n (rank 4 tensor) 
      
      dK_inv = -np.matmul(np.matmul(K_inv, dK), K_inv) # # 3 x n x n (rank 3 tensor)

      i, j = np.meshgrid(np.arange(3), np.arange(3))
      index = np.vstack([j.ravel(), i.ravel()]).T
      
      product1 = np.zeros(shape=(3,3,20,20))
      
      for i in index:
            print(i)
            product1[i[0]][i[1]] = np.matmul(dK_inv[i[0]], dK[i[1]])
            
      product2 = np.zeros(shape=(3,3,20,20))
      
      prel = np.matmul(K_inv, dK)
      
      for i in index:
            print(i)
            product2[i[0]][i[1]] = np.matmul(prel[i[0]], dK_inv[i[1]])
      
      d2K_inv = -np.matmul(product1, K_inv) - np.matmul(np.matmul(K_inv, d2K), K_inv) - product2
            
      vi_pred_mean_mf = []
      vi_pred_var_mf = []

      vi_pred_mean_fr = []
      vi_pred_var_fr = []
      
      vi_pred_mean_mf_mode = np.matmul(np.matmul(cov(X, X_star).T.eval(), K_inv), y)
      vi_pred_var_mf_mode =  np.diag(cov(X_star, X_star).eval() - np.matmul(np.matmul(cov(X, X_star).T.eval(), K_inv), cov(X, X_star).eval()))
                  
      for i in np.arange(len(X_star)):
          
          print(i)
          x_star = X_star[i].reshape(1,1)
          vi_pred_mean_mf.append(deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_mf, cov_theta_mf))
          #vi_pred_mean_fr.append(deterministic_gp_pred_mean(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_fr, cov_theta_fr))
          pred_var_mf_ = deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_mf, cov_theta_mf)
          print(pred_var_mf_ )
          vi_pred_var_mf.append(pred_var_mf_)
          #dt_pred_var_fr = deterministic_gp_pred_covariance(X, x_star, y, K, K_inv, K_noise, dK_inv, d2K_inv, mu_theta_fr, cov_theta_fr)
          #print(dt_pred_var_fr)
          #vi_pred_var_fr.append(dt_pred_var_fr)
          
        
        plt.figure()
        plt.plot(X_20, y_20, 'ko')
        plt.plot(X_star_20, post_means_fr_20.T, 'grey', alpha=0.4)
        plt.plot(X_star_20, f_star_20, 'k')
        plt.plot(X_star_20, pp_mean_fr_20, color='g')
        plt.plot(X_star_20, np.array(vi_pred_mean_fr).flatten(), color='b')
        plt.fill_between(X_star_20.ravel(), lower_fr_20, upper_fr_20, color='g', alpha=0.3)
        plt.fill_between(X_star_20.ravel(), (vi_pred_mean_mf - 1.96*np.sqrt(vi_pred_var_mf)).ravel(), (vi_pred_mean_mf + 1.96*np.sqrt(vi_pred_var_mf)).ravel(), alpha=0.5)
         