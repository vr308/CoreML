#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:00:03 2020

@author: vidhi

"""


import math
import numpy as np
import torch
import pymc3 as pm
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

def get_samples(kfunc, x):

    gp_prior = gpytorch.distributions.MultivariateNormal(mean=torch.tensor([0]*200), covariance_matrix=kfunc(x,x), validate_args=True)
    samples = gp_prior.sample(sample_shape=torch.Size([5]))
    return samples

def kernel_function(kfunc, x):
    tau = x - x[0]
    K = kfunc(x,x).evaluate()
    k_values = K[:,0]
    return tau, k_values.detach()


def kernel_matrix(kfunc, x):
    K = kfunc(x,x).evaluate()
    return K

def spectral_density(weights, means, scales):
    omega = np.linspace(min(means) - 5*max(scales), max(means) + 5*max(scales))
    gmm = pm.distributions.mixture.NormalMixture.dist(w=weights, mu=means,sigma=scales)
    pdf = np.exp(gmm.logp(omega)).eval()
    return omega, pdf


if __name__== "__main__":

    sm_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3, ard_num_dims=1)

    means = np.array([0.04, 0.09, 0.5])
    scales = np.array([0.01, 0.01, 0.01])
    weights = np.array([0.3,0.4,0.3])

    sm_kernel.mixture_means = means
    sm_kernel.mixture_scales = scales
    sm_kernel.mixture_weights = weights

    x = torch.linspace(0.01,10,200)

    #cosine_kernel_add = gpytorch.kernels.CosineKernel() + gpytorch.kernels.CosineKernel() + gpytorch.kernels.CosineKernel()
    #cosine_kernel.period_length = 1

    kfunc = sm_kernel

    K = kernel_matrix(sm_kernel, x)
    omega, pdf  = spectral_density(weights, means, scales)
    tau, k_values = kernel_function(kfunc, x)
    prior_samples_f = get_samples(kfunc, x)

    #plotting

    fig = plt.figure(figsize=(12,4))
    ax3 = fig.add_subplot(143)
    ax3.matshow(K.detach())
    ax1 = fig.add_subplot(141)
    ax1.plot(omega, pdf)

    ax2 = fig.add_subplot(142)
    ax2.plot(tau, k_values, color='green')
    ax2.set_xlabel('tau', fontsize='x-small')

    ax4 = fig.add_subplot(144)
    ax4.plot(x, prior_samples_f.T)


# pymc3 Spectral Mixture kernel - Additive product kernel


