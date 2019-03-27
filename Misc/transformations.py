#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:48:13 2019

@author: vidhi

Learning how probability transforms work

"""

import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-5, 10, 1000)
x_rv = st.norm.rvs(5,2, 1000)
y_rv = np.log(x_rv)

plt.figure()
plt.hist(x_rv, 100, normed=True)
plt.hist(y_rv, 100, normed=True)
plt.plot(x, st.norm.pdf(x, 5, 2))
plt.plot(x, st.norm.pdf(np.exp(x), 5, 2)*np.exp(x))

