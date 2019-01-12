# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

x = np.sort(np.random.uniform(-5,5,1000))
y = np.sort(np.random.uniform(-5,5,1000))

x_random = np.random.uniform(-5,5,1000)
y_random = np.random.uniform(-5,5,1000)

def linear_kernel(x,y,c):
    
    return x*y + c
    
def periodic(x,y,l,p):
    
    return np.exp(-2*np.square(np.sin(np.pi*(x-y)/p))/np.square(l))

def squared_exponential(x,l):
    
    return np.exp(-np.square(x-0)/(2*np.square(l)))

def rational_quadratic(x,y, l, alpha):
    
    return np.power((1 + np.square(x-y)/(2*alpha*np.square(l))),-alpha)

def brownian(x,y):
    
    mins = []
    for i,j in zip(x,y):
        mins.append(min(i,j))
    return mins


linPeriodic = linear_kernel(x,2,5)*periodic(x,1,1,2)
linSe = linear_kernel(x,y,5)*squared_exponential(x,1)
poly = linear_kernel(x,2,5)*linear_kernel(x,2,8)
perSe = periodic(x,1,1,1)*squared_exponential(x,1)

comb = squared_exponential(x,1) + linear_kernel(x,2,5)*periodic(x,1,1,2)

plt.figure(figsize=(10,4))
plt.subplot(131)
plt.plot(x, linPeriodic)
plt.title('Linear x Periodic', fontsize='small')
#plt.plot(x,linSe)
plt.subplot(132)
plt.plot(x, poly)
plt.title('Linear x Linear', fontsize='small')
plt.subplot(133)
plt.plot(x, perSe)
plt.title('Guassian x Periodic', fontsize='small')
#plt.plot(x, comb)

plt.figure(figsize=(12,4))
plt.subplot(141)
plt.plot(x, linear_kernel(x,2,5))
plt.title('Linear $k(x,y) = (x^{T}y + c)$', fontsize='small')
plt.subplot(142)
plt.plot(x, squared_exponential(x,1))
plt.title('Gaussian ' + r'$k(x,y) = \exp \lbrace \frac{-|x-y|^{2}}{2\sigma^{2}}\rbrace $', fontsize='small')

plt.subplot(143)
plt.plot(x, periodic(x,1,1,2))
plt.title('Periodic ' + r'$k(x,y) = (-\frac{2}{l^{2}}sin^{2}(\pi\frac{(x-y)}{p}))$', fontsize='small')

plt.subplot(144)
plt.plot(x, rational_quadratic(x,0,2,1))
plt.title('Rational Quadratic ' + r'$k(x,y) = (1 +  \frac{(x-y)^{2}}{2\alpha l^{2}})^{-\alpha}$', fontsize='small')
