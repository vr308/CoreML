#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:52:00 2020

@author: vidhi
"""

import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Visualising kernel matrices

x = torch.linspace(0.01,10,200)

lin_covar = gpytorch.kernels.LinearKernel(num_dimensions=1)
poly_covar = gpytorch.kernels.PolynomialKernel(power=4)

se_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
se_covar.lenghtscale = 5
se_covar.outputscale = 10

per_covar = gpytorch.kernels.PeriodicKernel()
per_covar.lenghtscale = 3
per_covar.period_length = 2

rq_covar = gpytorch.kernels.RQKernel()
rq_covar.lengthscale = 2
rq_covar.alpha = 10

# Creating the kernel matrix

K_lin = lin_covar(x,x).evaluate()
K_poly = poly_covar(x,x).evaluate()
K_se = se_covar(x,x).evaluate()
K_per = per_covar(x,x).evaluate()
K_rq = rq_covar(x,x).evaluate()

# Plotting

plt.matshow(K_lin.detach())
plt.matshow(K_poly.detach())
plt.matshow(K_se.detach())
plt.matshow(K_per.detach())
plt.matshow(K_rq.detach())


# Drawing functions from kernels with fixed hyperparameters (Forward sampling)

def get_samples(covar):

    gp_prior = gpytorch.distributions.MultivariateNormal(mean=torch.tensor([0]*200), covariance_matrix=covar, validate_args=True)
    samples = gp_prior.sample(sample_shape=torch.Size([10]))
    return samples

se_samples = get_samples(se_covar(x,x))
per_samples = get_samples(per_covar(x,x))
rq_samples = get_samples(rq_covar(x,x))

plt.figure(figsize=(15,8))
plt.subplot(231)
plt.plot(x, se_samples.T, color='grey')

plt.subplot(232)
plt.plot(x, per_samples.T,color='grey')

plt.subplot(233)
plt.plot(x, rq_samples.T, color='grey')


# Drawing functions from kernels with hyper priors (Forward sampling with hyper priors)

os_prior = gpytorch.priors.LogNormalPrior(10,10)
l_prior = gpytorch.priors.GammaPrior(20,0.5)

se_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=l_prior), outputscale_prior = os_prior)

se_samples = get_samples(se_covar(x,x))

per_samples = get_samples(per_covar(x,x))
rq_samples = get_samples(rq_covar(x,x))


plt.subplot(234)
plt.plot(x, se_samples.T)

plt.subplot(235)
plt.plot(x, per_samples.T,color='grey')

plt.subplot(236)
plt.plot(x, rq_samples.T, color='grey')

