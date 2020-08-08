#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

import numpy as np
import pymc3 as pm 
import matplotlib.pylab as plt
import tensorflow as tf

X = np.linspace(-10, 10, 1000)[:,None]

lengthscale = 2
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

K = cov(X).eval()

plt.matshow(K)

