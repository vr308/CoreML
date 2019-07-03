#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:34:27 2019

@author: vidhi

forward mode and reverse mode differentiation for covariance matrix

"""

import torch
import gpytorch
import numpy as np
from autograd import grad, elementwise_grad
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

# Evaluating the gradient at multiple xy points 

# TODO 

x = np.linspace(0, 1, 11)
y = 0.0

x, y = np.broadcast_arrays(x, y)
grad_func(x,y)


f = lambda x,y: (x+y)**2
fgrad = grad(f, 0)

x = np.linspace(0, 1, 11)
y = 0.0

x, y = np.broadcast_arrays(x, y)

# Function in a matrix

# Generate a standard covariance matrix to double check answers
import pymc3 as pm
import theano.tensor as tt

sig_sd = 1
ls = 1
noise_sd = 1
    
mean = pm.gp.mean.Zero()
cov = pm.gp.cov.Constant(sig_sd**2)*pm.gp.cov.ExpQuad(1, ls=ls)
f = np.random.multivariate_normal(mean(X).eval(), cov=cov(X, X).eval())

y = f + np.random.normal(0,1, 100)
X = np.linspace(1,5, 100).reshape(100,1)
X_star = np.linspace(1,5,25).reshape(25,1)

x_star = np.array([4]).reshape(1,1)

K = cov(X).eval()
K_s = cov(X, x_star).eval()

def se_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      if x1 != x2:
            return sig_sd**2*np.exp(-0.5*(1/ls**2)*(x1 - x2)**2)
      else: 
            return sig_sd**2*np.exp(-0.5*(1/ls**2)*(x1 - x2)**2) + noise_sd**2
      
def kernel_matrix(sig_sd, ls, noise_sd, *args):
            
   X = args[0]
   n_train = len(X)
 
   if len(args) == 1:    
         K = np.zeros(shape=(n_train,n_train))
         i, j = np.meshgrid(np.arange(n_train), np.arange(n_train))
         index = np.vstack([j.ravel(), i.ravel()]).T
            
         for h in index:
            K[h[0]][h[1]] = se_kernel(sig_sd, ls, noise_sd, X[h[0]], X[h[1]])
         return K
   else:
         x_star = args[1][0]
         n_test = len(x_star)
         K_s = np.zeros(shape=(n_train, n_test))
         
         for h in np.arange(n_train):
               print(h)
               K_s[h] = se_kernel(sig_sd, ls, noise_sd, X[h], x_star)
         return K_s

def h_theta(sig_sd, ls, noise_sd, X, y, x_star):
      
      K_noise = kernel_matrix(sig_sd, ls, noise_sd, X)
      K_s = kernel_matrix(sig_sd, ls, noise_sd, X, x_star)
      return np.matmul(np.matmul(K_s.T, np.linalg.inv(K_noise)), y)

grad_omg = grad(h_theta, argnum=[0,1,2])
deriv = grad_omg(1.0,1.0,1.0, X, y, x_star)


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