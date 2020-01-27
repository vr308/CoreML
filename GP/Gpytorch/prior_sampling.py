#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:52:00 2020

@author: vidhi
"""

import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

# Visualising kernel matrices

x = torch.linspace(0.01,10,200)

lin_covar = gpytorch.kernels.LinearKernel(num_dimensions=1)
poly_covar = gpytorch.kernels.PolynomialKernel(power=1)
se_covar = gpytorch.kernels.RBFKernel()
per_covar = gpytorch.kernels.PeriodicKernel()
rq_covar = gpytorch.kernels.RQKernel()

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





# Drawing functions from kernels with hyper priors (Forward sampling with hyper priors)







