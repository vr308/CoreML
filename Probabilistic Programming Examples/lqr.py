#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:49:43 2019

@author: vidhi
"""

import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import ArtistAnimation
import seaborn as sns
import scipy as sp
import theano.tensor as tt
import scipy.stats as st
from matplotlib.patches import Ellipse
import warnings 

warnings.filterwarnings('ignore')

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))


X = np.array([-1.3, -0.7, -0.1, -0.05])[:,None]
y = np.array([3.5, 2.5, 3.0, 4.0])

with pm.Model() as lqr_model:
      
      a = pm.Uniform('a', lower= 0.5, upper = 1.5)
      b = pm.Uniform('b', lower= 0.5, upper = 1.5)
      sigma_w = pm.Uniform('sigma_w', lower=0, upper=100)
      
      k = pm.gp.cov.Constant(sigma_w**2)*LQR(1,1, a, b)
      
      gp = pm.gp.Marginal(cov_func=k)
            
      # Marginal Likelihood
      y_ = gp.marginal_likelihood("y", X=X, y=y, noise=0.001)
       
with lqr_model:
      
      posterior = pm.sample(5000)
      f_cond = gp.conditional("f_cond", Xnew=X_star)

      
class LQR(pm.gp.cov.Covariance):
    def __init__(self, q, r, a, b):
        super(LQR, self).__init__(1, None)
        self.q = q
        self.r = r
        self.a = a
        self.b = b

    def kernel_func(self, X):
          
          return (self.q + self.r*X**2)/(1 - (self.a+self.b*X)**2)

    def diag(self, X):
        return tt.alloc(tt.square(self.kernel_func(X)), X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return tt.alloc(tt.diag(tt.square(self.kernel_func(X))),X.shape[0],X.shape[0])
        else:
            return tt.alloc(tt.dot(self.kernel_func(X), tt.transpose(self.kernel_func(Xs))),X.shape[0],X.shape[0])
      
class LQR(pm.gp.cov.Covariance):
    def __init__(self, q, r, a, b):
        super(LQR, self).__init__(1, None)
        self.q = q
        self.r = r
        self.a = a
        self.b = b

    def kernel_func(self, X):
          
          return (self.q + self.r*X**2)/(1 - (self.a+self.b*X)**2)

    def diag(self, X):
        return tt.square(self.kernel_func(X))

    def full(self, X, Xs=None):
        if Xs is None:
            return self.kernel_func(X).__mul__(self.kernel_func(X))
        else:
            return self.kernel_func(X).__mul__(self.kernel_func(Xs))
      
k = LQR(1, 1, 1, 1)

x = np.array([-0.5, 0.5])
xs = np.array([-0.5, 0.5])

k.full(x)


class WhiteNoise(pm.gp.cov.Covariance):
    def __init__(self, sigma):
        super(WhiteNoise, self).__init__(1, None)
        self.sigma = sigma

    def diag(self, X):
        return tt.alloc(tt.square(self.sigma), X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return tt.diag(self.diag(X))
        else:
            return tt.alloc(0.0, X.shape[0], Xs.shape[0])
      
k = WhiteNoise(sigma=1)

x = np.array([-0.5, 0.5])
xs = np.array([-0.5, 0.5])

