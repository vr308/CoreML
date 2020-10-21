#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:52:06 2020

@author: vr308
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import scipy as sp

x = np.linspace(0.0,5,100)
lam = 7
mu = 1
sig_f = 5
l = 2
alpha = 0.01

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

kf = integrated_spectral_mixture(x, sig_f, lam, mu, l, alpha)
kfu = integrated_spectral_mixture_uniform(x, sig_f, 3, 2, l, alpha)

columns = np.empty((100,100))
for i in np.arange(100):
    columns[i] = integrated_spectral_mixture(x, sig_f, lam, mu, l, alpha)

K = sp.linalg.toeplitz(kfu)
plt.matshow(columns)