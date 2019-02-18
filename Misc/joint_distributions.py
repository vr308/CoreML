#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vidhi

Specifying exotic joint distributions as a test bed for sampler testing 

"""

import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import scipy as sp
from matplotlib import cm
import pymc3 as pm
import seaborn as sns
from scipy.stats import rv_continuous
plt.style.use("ggplot")

# Mixture of  bivariate normals

x_ = np.linspace(-5,5, 1000)
density_1 = lambda x_ : 0.4*st.norm.pdf(x_, 2, 1) + 0.6*st.norm.pdf(x_, -1, 1)
density_2 = lambda x_ : st.norm.pdf(x_,4,0.3)  

joint_density = lambda x1_, x2_ : density_1(x1_)*density_2(x2_)

# Check that this joint density construction is a valid one

sp.integrate.dblquad(joint_density, -5, +5, lambda x: -5, lambda x: +5)

X1, X2 = np.meshgrid(x_,x_)
Y = joint_density(X1,X2)
plt.figure()
plt.contourf(X1,X2,Y, cmap=cm.get_cmap('jet'),alpha=0.5)
plt.ylim(2.5,5)
plt.xlim(-6, 6)

# Testing samples from seaborn construction 

k=np.random.choice([1,2], size=1000, p = [0.4,0.6])

x1 = []

for i in k:
      if i == 1:
            x1.append(st.norm.rvs(2,1))
      elif i == 2:
            x1.append(st.norm.rvs(-1,1))

x2 = np.random.normal(4, 0.3, 1000)

plt.hist(x1, bins=100, normed=True)
plt.plot(x_, density_1(x_))

sns.jointplot(x1, x2, kind='kde')

#---------------------------------------------------------------------

# Donut / Ring distribution

#---------------------------------------------------------------------






#----------------------------------------------------------------------------
# Banana shaped distribution 

# It is obtained by applying a simple transformation to a bi-variate Gaussian
#----------------------------------------------------------------------------

mean = [1,2]
cov = np.array(([1, 0.7], [0.7, 1]))
x_ = np.random.multivariate_normal(mean, cov, size=1000)

def b_transformation(x_):
      
      column_1 = x_[:,0]
      column_2 = x_[:,1] - np.square(x_[:,0]) -1 
      return np.asarray((column_1, column_2)).T.reshape(1000,2)
      
y_ = b_transformation(x_)     

# The transformed samples do indeed look like they have come from a banana shaped dist.
  
sns.jointplot(y_[:,0], y_[:,1], kind='kde')

# Joint density of the banana shaped posterior

joint_density = lambda x1_, x2_ : density_1(x1_)*density_2(x2_)

joint = []

for i in np.arange(1000):
      joint.append(st.multivariate_normal.pdf(y_[i], mean, cov))







