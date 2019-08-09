#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:27:17 2019

@author: vidhi

Visualising GP samples from priors of composite kernels

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# Priors on Kernel hyp

x = np.linspace(0, 150, 5000)


priors = [
    ("ℓ_pdecay",  pm.Gamma.dist(alpha=10, beta=0.075)),
    ("ℓ_psmooth", pm.Gamma.dist(alpha=4,  beta=3)),
    ("period",    pm.Normal.dist(mu=1.0,  sd=0.05)),
    ("ℓ_med",     pm.Gamma.dist(alpha=2,  beta=0.75)),
    ("α",         pm.Gamma.dist(alpha=5,  beta=2)),
    ("ℓ_trend",   pm.Gamma.dist(alpha=4,  beta=0.1)),
    ("ℓ_noise",   pm.Gamma.dist(alpha=2,  beta=4))]

colors = brewer['Paired'][7]

plt.figure()
plt.ylabel('Probability')
plt.xlabel('Years')
plt.title('Lengthscale and period priors')

for i, prior in enumerate(priors):
    plt.plot(x, np.exp(prior[1].logp(x).eval()),
           lw=1, color=colors[i], label=priors[i][0])
plt.legend()


X = np.linspace(0,100,1000)[:,None]

with pm.Model() as composite_priors:
      
      dim = 1
      
      # Mean function
      
      c = 0
      
      mean_func_lin = pm.gp.mean.Linear(coeffs=c, intercept=1)
      mean_func_zero = pm.gp.mean.Zero()
      mean_func_const = pm.gp.mean.Constant(c=3)
      
      # hyp poly
      
      c_poly=1
      
      # hyp lin
      
      c_lin = 9
      
      # hyp per
      
      s_per = 1
      ls_per = 2
      
      # hyp mat
      
      s_mat = 10
      ls_mat = 5
      
      # hyp rq
      
      alpha_rq = 0.01
      ls_rq = 2
      
      # hyp se 
      
      s_se = 1
      ls_se = 20

      # hyp noise
      
      n_sd = 10

      k_lin = pm.gp.cov.Linear(1, c=c_lin)
      k_poly = pm.gp.cov.Polynomial(1, c=c_poly, d=2, offset=0)
      k_se = pm.gp.cov.Constant(s_se**2)*pm.gp.cov.ExpQuad(1, ls_se)
      k_mat_52 = pm.gp.cov.Constant(s_mat**2)*pm.gp.cov.Matern52(1, ls_mat) 
      k_mat_32 = pm.gp.cov.Constant(s_mat**2)*pm.gp.cov.Matern32(1, ls_mat) 
      k_rq = pm.gp.cov.RatQuad(1, alpha=alpha_rq, ls=ls_rq)
      k_per = pm.gp.cov.Constant(s_per**2)*pm.gp.cov.Periodic(1, period=p, ls=ls_per)
      k_noise = pm.gp.cov.WhiteNoise(n_sd**2)
      

      k = k_mat_32 + k_se
          
      gp = pm.gp.Latent(mean_func = mean_func_zero, cov_func=k)
      
      #Prior 
      f = gp.prior('f', X)
      
      
with composite_priors:
      
      trace = pm.sample(tune=5,draws=20)


# Plotting
    
plt.figure()
plt.plot(X, trace['f'].T[:,0:10], color='gray')
