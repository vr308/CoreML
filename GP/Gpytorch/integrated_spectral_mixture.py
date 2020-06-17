#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:52:06 2020

@author: vr308
"""

import numpy as np
import math 

x = np.linspace(-1,1,100)
lam = 7
mu = 1
sig_f = 5
l = 2
alpha = 0.01

def integrated_spectral_mixture(x, x0, sig_f, lam, mu, l, alpha):
    tau = x - x0
    pi_term = 2*np.pi*tau*mu
    mu_int_term = np.divide(np.exp(-lam*mu), lam**2 + 4*(np.pi**2)*tau**2)*(-lam*np.cos(pi_term) + 2*np.pi*tau*np.sin(pi_term))
    scale_term = np.power(1 + np.divide(2*(np.pi**2)*(tau**2), alpha*(l**2)), -alpha)
    return -sig_f**2*(mu_int_term)*(scale_term)

kf = integrated_spectral_mixture(x, x[0], sig_f, lam, mu, l, alpha)

columns = np.empty((100,100))
for i in np.arange(100):
    columns[i] = integrated_spectral_mixture(x, x[i], sig_f, lam, mu, l, alpha)
    
plt.matshow(columns)