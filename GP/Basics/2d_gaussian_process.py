#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:07:11 2019

@author: vidhi
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
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

xv, yv = torch.meshgrid([torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)])

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
bool_idx = np.array([i in train_index for i in arr])
test_index = arr[~bool_idx]

#fig = plt.figure(figsize=(10,6))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xv, yv, train_y.reshape(100,100).numpy(), cmap=cm.jet, alpha=0.4)

plt.figure()
plt.contourf(xv, yv, y.reshape(100,100).numpy(), levels=50, cmap=cm.jet)
plt.plot(X[train_index].numpy()[:,0], X[train_index].numpy()[:,1],'kx')


# Creating test and train data

train_x = X[train_index]
test_X = X[test_index]

train_y = y[train_index]
test_y = y[test_index]

# ML-II

# initialize likelihood and model

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
{'params': model.parameters()}, # Includes GaussianLikelihood parameters
], lr=0.1)

 # "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = model(train_x)
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
      
