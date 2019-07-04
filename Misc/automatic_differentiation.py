#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:34:27 2019

@author: vidhi

forward mode and reverse mode differentiation for covariance matrix

"""

import torch
import gpytorch
from autograd import grad, elementwise_grad, jacobian, hessian
import autograd.numpy as np
from sympy import symbols, diff, exp, log, power, sin, cos

# Simple function of two variables

x, y = symbols('x y')

r = np.array([0.0,0.0])

def exp_comp_ad(x, y):                 # Define a function
     return np.exp(np.sin(x)*np.log(y))

def exp_comp_sym(x, y):                 # Define a function
     return exp(sin(x)*log(y))

def symbolic_grad(x,y):
      
      da = np.cos(x)*np.log(y)
      db = np.sin(x)/y
      c =  np.exp(np.sin(x)*np.log(y))
      return [c*da,  c*db]


diff(exp_comp_sym(x, y), x).subs({x : 1, y:2})
diff(exp_comp_sym(x, y), y).subs({x : 1, y:2})

grad_func = grad(exp_comp_ad, argnum=[0,1])

#Should match 
grad_func(1.0,2.0)
symbolic_grad(1,2)

# Logistic regression example

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
print("Initial loss:", training_loss(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))

# Differentiating the GP mean wrt theta

# Generate a standard covariance matrix to double check answers

import pymc3 as pm
import theano.tensor as tt

sig_sd = 1.0
ls = 1.0
noise_sd = 1.0
    
mean = pm.gp.mean.Zero()
cov = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
f = np.random.multivariate_normal(mean(X).eval(), cov=cov(X, X).eval())

y = f + np.random.normal(0,1, 100)
X = np.linspace(1,5, 100).reshape(100,1)
X_star = np.linspace(1,5,25).reshape(25,1)

x_star = np.array([4]).reshape(1,1)

K = cov(X).eval()
K_s = cov(X, x_star).eval()
K_noise = K + noise_sd**2 * np.eye(len(X))
K_inv = np.linalg.inv(K_noise)

# Derivative of K^-1

def get_K_inv(sig_sd, ls, noise_sd, X):
      
      K = kernel(sig_sd, ls, noise_sd, X, X)
      K_noise = K + noise_sd**2 * np.eye(len(X))
      return np.linalg.inv(K_noise)

grad_kinv = jacobian(get_K_inv, argnum=0)
deriv_kinv = grad_kinv(1.0,1.0,1.0, X)

def get_K(sig_sd, ls, noise_sd, X):
      
      K = kernel(sig_sd, ls, noise_sd, X, X)
      K_noise = K + noise_sd**2 * np.eye(len(X))
      return K_noise

grad_k = jacobian(get_K, argnum=[0,1])
deriv_k = grad_k(1.0,1.0,1.0, X)

theta = np.array([sig_sd, ls, noise_sd])

def kernel(theta, X1, X2):
    ''' Isotropic squared exponential kernel. 
    Computes a covariance matrix from points in X1 and X2. 
    Args: X1: Array    of m points (m x d). X2: Array of n points (n x d). 
    Returns: Covariance matrix (m x n). '''
    
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return theta[0]**2 * np.exp(-0.5 / theta[1]**2 * sqdist)

def gp_mean(theta, X, y, x_star):
      
      K = kernel(theta, X, X)
      K_noise = K + theta[2]**2 * np.eye(len(X))
      K_s = kernel(theta, X, x_star)
      return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

dh = elementwise_grad(gp_mean)
d2h = jacobian(dh)
#d2hh = hessian(gp_mean)

deriv = dh(theta, X, y, x_star)
deriv2 = d2h(theta, X, y, x_star)

# Symbolic cross checking

from sympy import symbols, diff, exp, log, power, sin

sig_sd, ls, noise_sd, x1, x2 = symbols('sig_sd ls noise_sd x1 x2', real=True)

def se_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*exp(-0.5*(1/ls**2)*(x1 - x2)**2) + noise_sd**2

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

def gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s):
      
      dK_starT = gradient_K_star(X, x_star).T
      
      return np.matmul(np.matmul(dK_starT, K_inv), y)[:, np.newaxis] + np.matmul(np.matmul(K_s.T, dK_inv), y)

def curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv):
      
      dK_starT = gradient_K_star(X, x_star).T
      d2K_star = curvature_K_star(X, x_star)
      d2K_starT = np.array([d2K_star[0].T, d2K_star[1].T, d2K_star[2].T])
      
      return  np.matmul(np.matmul(d2K_starT, K_inv), y) + 2*np.matmul(np.matmul(dK_starT, dK_inv), y) + np.matmul(np.matmul(K_s.T, d2K_inv),y).reshape(3,3).T


dk_dsf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd).subs({sig_sd:1, ls: 1, noise_sd: 1})
dk_dls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls).subs({sig_sd:1, ls: 1, noise_sd: 1})
dk_dsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd).subs({sig_sd:1, ls: 1, noise_sd: 1})

d2k_d2sf = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, 2).subs({sig_sd:1, ls: 1, noise_sd: 1})
d2k_d2ls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, 2).subs({sig_sd:1, ls: 1, noise_sd: 1})
d2k_d2sn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), noise_sd, 2).subs({sig_sd:1, ls: 1, noise_sd: 1})

d2k_dsfdls = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, ls).subs({sig_sd:1, ls: 1, noise_sd: 1})
d2k_dsfdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), sig_sd, noise_sd).subs({sig_sd:1, ls: 1, noise_sd: 1})
d2k_dlsdsn = diff(se_kernel(sig_sd, ls, noise_sd, x1, x2), ls, noise_sd).subs({sig_sd:1, ls: 1, noise_sd: 1})
      
i, j = np.meshgrid(np.arange(3), np.arange(3))
index = np.vstack([j.ravel(), i.ravel()]).T

tensor_mult1 = np.zeros(shape=(3,3,len(X),len(X)))

for i in index:
      tensor_mult1[i[0]][i[1]] = np.matmul(dK_inv[i[0]], dK[i[1]])
      
tensor_mult2 = np.zeros(shape=(3,3,len(X),len(X)))
prel = np.matmul(K_inv, dK)

for i in index:
      tensor_mult2[i[0]][i[1]] = np.matmul(prel[i[0]], dK_inv[i[1]])

dK = gradient_K(X) # 3 x n x n (rank 3 tensor)
d2K = curvature_K(X) # 3 x 3 x n x n (rank 4 tensor) 
dK_inv = -np.matmul(np.matmul(K_inv, dK), K_inv) # # 3 x n x n (rank 3 tensor)
d2K_inv = -np.matmul(tensor_mult1, K_inv) - np.matmul(np.matmul(K_inv, d2K), K_inv) - tensor_mult2

deriv_analytical = gradient_gp_pred_mean(X, x_star, y, K_inv, dK_inv, K_s)
deriv2_analytical = curvature_gp_pred_mean(X, x_star, y, K_s, K_inv, dK_inv, d2K_inv)