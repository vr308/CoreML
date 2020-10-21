#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:52:06 2020

@author: vr308

"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import torch
import gpytorch

def get_samples(K, x):

    gp_prior = gpytorch.distributions.MultivariateNormal(mean=torch.tensor([0]*200), covariance_matrix=K, validate_args=True)
    samples = gp_prior.sample(sample_shape=torch.Size([5]))
    return samples

def integrated_spectral_mixture(tau, sig_f, lam, mu, l, alpha):
    #pi_term = 2*np.pi*tau*mu
   # mu_int_term = np.divide(np.exp(-lam*mu), lam**2 + 4*(np.pi**2)*tau**2)*(-lam*np.cos(pi_term) + 2*np.pi*tau*np.sin(pi_term))
    int_term = (lam*sig_f)**2*(1/(lam**2 + (2*np.pi*tau)**2))
    scale_term = np.power(1 + np.divide(2*(np.pi**2)*(tau**2), alpha*(l**2)), -alpha)
    return int_term*scale_term

def integrated_spectral_mixture_uniform(tau, sig_f, b, a, l, alpha):
    const_term = 1/(2*np.pi*(b-a))*sig_f**2
    int_term = (np.sin(2*np.pi*tau*b) - np.sin(2*np.pi*tau*a))/tau
    int_term[tau == 0] = 2*np.pi*(b-a)
    rq_term = np.power(1 + np.divide(2*(np.pi**2)*(tau**2), alpha*(l**2)), -alpha)
    return const_term*int_term*rq_term

if __name__== "__main__":
    
    x = np.linspace(0.0,5,100)
    lam = 7
    mu = 1
    sig_f = 5
    l = 2
    alpha = 0.01
    
    # Prior pdf
    
    
    # Kernel function (tau)
    
    kf = integrated_spectral_mixture(x, sig_f, lam, mu, l, alpha)
    kfu = integrated_spectral_mixture_uniform(x, sig_f, 3, 2, l, alpha)

    # Kernel matrix 
    
    sp.linalg.toeplitz
    
    K = sp.linalg.toeplitz(kf)
    K_fu= sp.linalg.toeplitz(kfu)

    #plotting

    fig = plt.figure(figsize=(12,4))
  
    ax1 = fig.add_subplot(141)
    ax1.plot(omega, pdf)
    ax1.set_title(r'Prior on \mu', fontsize='x-small')

    ax2 = fig.add_subplot(142)
    ax2.plot(x, kf, color='green')
    ax2.set_xlabel('tau', fontsize='x-small')
    ax2.set_title('Kernel Function', fontsize='x-small')
    
    ax3 = fig.add_subplot(143)
    ax3.matshow(K.detach())
    ax3.set_title('Kernel Matrix', fontsize='x-small')
    ax3.tick_params(axis="x", labelsize=8)
    ax3.tick_params(axis="y", labelsize=8)

    ax4 = fig.add_subplot(144)
    ax4.plot(x, prior_samples_f.T)
    ax4.set_title('Functions drawn from the prior', fontsize='x-small')

