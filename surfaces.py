#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:38:51 2018

@author: vr308
"""

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as intp
from matplotlib import cm
import random
import numpy as np

x = y = np.linspace(-3,3, 1000)
X, Y = np.meshgrid(x,y)
Z = np.square(np.sin(X))*np.cos(Y)

x_sample = random.sample(x,100)
y_sample = random.sample(y,100)
z_sample = np.square(np.sin(x_sample))*np.cos(y_sample) + 0.2*np.random.randn(100)

# 3D rendering of the function along with scatter
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_sample,y_sample,z_sample, 'b^')
ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('spring'), alpha=0.5)

# 2D contour of the mathematical function along with scatter

plt.figure()
plt.contourf(X,Y,Z, cmap=cm.get_cmap('spring'))
plt.plot(x_sample,y_sample,'bo')

# Multivariate interpolation using rbf technique applied to scatter data

rbf_interp = intp.Rbf(x_sample,y_sample,z_sample,smooth=1)
Z_interp = rbf_interp(X,Y)

# Multivariate interpolation using nearest neighbour technique applied to scatter data

inputs = np.array([x_sample,y_sample]).reshape(100,2)
outputs = z_sample
ninterp = intp.NearestNDInterpolator(inputs, outputs)

Z_interp = ninterp(X,Y)


# Plot the interpolated surface

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_sample,y_sample,z_sample, 'b^')
ax.plot_surface(X,Y,Z_interp, cmap=cm.get_cmap('spring'), alpha=0.5)

# Plot the contour 
plt.figure()
plt.contourf(X,Y,Z, cmap=cm.get_cmap('spring'))
plt.plot(x_sample,y_sample,'bo')

plt.figure()
plt.contourf(X,Y,Z_interp, cmap=cm.get_cmap('spring'))
plt.plot(x_sample,y_sample,'bo')


