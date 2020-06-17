#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:38:24 2020

@author: vr308

Understanding gradients in pytorch

"""

import torch
from torch.autograd import Variable
import gpytorch

x = Variable(torch.ones(10), requires_grad=True)
y = x**2 
y.backward(torch.ones(10))
print(x.grad)

sig_f = Variable(1, requires_grad=True)
ls = Variable(1, requires_grad=True)

se_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())