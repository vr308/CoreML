#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:07:11 2019

@author: vidhi
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from mpl_toolkits.mplot3d import axes3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib import cm
import torch
import gpytorch 
from matplotlib import pyplot as plt

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
      
      def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
      def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def franke(X, Y):
      term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
      term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
      term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
      term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))
      f = term1 + term2 + term3 - term4
      #dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
      #-2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
      #dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
      #-2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4
      return f

# Creating some data 

xv, yv = torch.meshgrid([torch.linspace(0, 1, 10), torch.linspace(0, 1, 10)])

X = torch.cat((
xv.contiguous().view(xv.numel(), 1),
yv.contiguous().view(yv.numel(), 1)),
dim=1
)
      
f = franke(X[:, 0], X[:, 1])
y = torch.stack([f], -1).squeeze(1)
y += 0.05 * torch.randn(y.size()) # Add noise to both values and gradients

arr = np.arange(len(X))
train_index = np.random.randint(0,len(X),40)
#bool_idx = np.array([i in train_index for i in arr])
#test_index = arr[~bool_idx]

# Creating test and train data

train_X = X[train_index]
train_y = y[train_index]

# Creating a new grid for test
xv, yv = torch.meshgrid([torch.linspace(0, 1, 120), torch.linspace(0, 1, 120)])

test_X = torch.cat((
xv.contiguous().view(xv.numel(), 1),
yv.contiguous().view(yv.numel(), 1)),
dim=1)

f = franke(test_X[:, 0], test_X[:, 1])
test_y = torch.stack([f], -1).squeeze(1)

plt.figure()
plt.contourf(xv, yv, f.reshape(120,120).numpy(), levels=50, cmap=cm.jet)
plt.plot(X[train_index].numpy()[:,0], X[train_index].numpy()[:,1],'kx')

# ML-II

# initialize likelihood and model

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_X, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
{'params': model.parameters()}, # Includes GaussianLikelihood parameters
], lr=0.1)

 # "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 200
for i in range(training_iter):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = model(train_X)
      # Calc loss and backprop gradients
      loss = -mll(output, train_y)
      loss.backward()
      print('Iter %d/%d - Loss: %.3f signal_sd: %.3f lengthscale: %.3f noise_sd: %.3f' % (
      i + 1, training_iter, loss.item(),
      model.covar_module.outputscale.item(),
      model.covar_module.base_kernel.lengthscale.item(),
      model.likelihood.noise.item()
      ))
      optimizer.step()
      
      
# torch predictions
      

      
# ML-II sklearn
      
# sklearn kernel 

# se-ard + noise

se_ard = Ck(10.0)*RBF(length_scale=np.array([1.0]*2), length_scale_bounds=(0.000001,1e5))
     
noise = WhiteKernel(noise_level=1**2,
            noise_level_bounds=(1e-5, 100))  # noise terms

sk_kernel = se_ard + noise

gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=5)
gpr.fit(train_X, train_y)

mu_test, std_test = gpr.predict(test_X, return_std=True)

plt.figure(figsize=(10,6))
plt.subplot(121)
plt.contourf(xv, yv, f.reshape(120,120), levels=50, cmap=cm.jet)
plt.plot(train_X.numpy()[:,0], train_X.numpy()[:,1],'kx')
plt.subplot(122)
plt.contourf(xv, yv, mu_test.reshape(120,120), levels=50, cmap=cm.jet)
plt.plot(train_X.numpy()[:,0], train_X.numpy()[:,1],'kx')

# HMC
      
with pm.Model() as dmodel:
     
     log_s = pm.Normal('log_s', 0, 3)
     log_ls = pm.Normal('log_ls', mu=np.array([0]*2), sd=np.ones(2,)*3, shape=(2,))
     log_n = pm.Normal('log_n', 0, 3)
     
     s = pm.Deterministic('s', tt.exp(log_s))
     ls = pm.Deterministic('ls', tt.exp(log_ls))
     n = pm.Deterministic('n', tt.exp(log_n))
     
     # Specify the covariance function
 
     cov_main = pm.gp.cov.Constant(s**2)*pm.gp.cov.ExpQuad(2, ls)
     cov_noise = pm.gp.cov.WhiteNoise(n**2)
 
     gp_main = pm.gp.Marginal(cov_func=cov_main)
     gp_noise = pm.gp.Marginal(cov_func=cov_noise) 
     
     gp = gp_main
     
     trace_prior = pm.sample(500)
     
with dmodel:
      
     # Marginal Likelihood
     y_ = gp.marginal_likelihood("y", X=train_X.numpy(), y=train_y.numpy(), noise=cov_noise)
 
with dmodel:

      trace_hmc = pm.sample(draws=1000, tune=500, chains=2)
      
      
trace_prior_df = pm.trace_to_dataframe(trace_prior)
trace_hmc_df = pm.trace_to_dataframe(trace_hmc)      
