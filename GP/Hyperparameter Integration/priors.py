#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:04:15 2019

@author: vidhi
"""

import matplotlib.pylab as plt
import numpy as np

# Potential Priors

plt.figure(figsize=(12,8))

plt.suptitle('Potential Priors')

x = np.linspace(0,10,1000)

# Gamma

plt.subplot(221)

params = [(1,2), (2,2), (3,2), (5, 1)]

for j in params:
      print(j)
      plt.plot(x, st.gamma.pdf(x, j[0], scale=1/j[1]), label=str(j))
plt.legend()
plt.title('Gamma')

# Half Normal

plt.subplot(222)

params = [(0.1,1), (0.1,2), (0.1,4)]

for j in params:
      print(j)
      plt.plot(x, st.halfnorm.pdf(x, j[0],j[1]), label=str(j))
plt.legend()
plt.title('Half-normal')

# Half Cauchy

plt.subplot(223)

params = [(0.1,1), (0.1,2), (0.1,4)]

for j in params:
      print(j)
      plt.plot(x, st.halfcauchy.pdf(x, j[0],j[1]), label=str(j))
plt.legend()
plt.title('Half-Cauchy')

# Log-normal 

plt.subplot(224)

params = [(0,0.25), (0,0.5), (0,1)] #(1,0.5),(1,1), (1,2)]

for j in params:
      print(j)
      plt.plot(x, st.lognorm.pdf(x, j[1],j[0], 1), label=str(j))
plt.legend()
plt.title('Log-Normal')
