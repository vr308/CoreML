#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vidhi
"""

import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import pymc3 as pm
import seaborn as sns
plt.style.use("ggplot")


def donut_pdf(scale):
    """Sample pdf for visualizations.  
    
    Normally distributed around the unit 
    circle, and the radius may be scaled.
    """
    def logp(x):
        return -((1 - np.linalg.norm(x)) / scale) ** 2
    return logp


x1 = np.random.normal(1,2, size=1000)
x2 = np.random.normal(5,4, size=1000)

x3 = np.concatenate([x1,x2])
x4 = np.random.beta(1,4, 2000)

sns.jointplot(x3, x4, kind='kde')


y = np.sin(x1)

sns.jointplot(x1, y, kind='kde')

x_range = np.linspace(-7.5, 7.5, 1000)
y_range = np.linspace(-1.5, 1.5, 1000)

z_ = np.meshgrid(x_range, y_range)

