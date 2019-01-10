#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:38:51 2018

@author: vr308
"""

import matplotlib.pylab as plt
import scipy.interpolate as intp
from matplotlib import cm
import random
import numpy as np

#np.square(np.sin(X))*np.cos(Y) + Y*np.sin(X)
#X*np.exp(-np.square(X) - np.square(Y))

def sample_surface(X,Y):
    
    return((np.subtract(X,1)*np.exp(-np.square(np.subtract(X,1)) - np.square(Y)) + np.cos(X)*np.exp(-np.square(np.add(Y,1)) - np.square(X)) + np.exp(-np.square(np.add(X,2)) - np.square(np.add(Y,-2)))))

x = y = np.linspace(-4,4,1000)
X, Y = np.meshgrid(x,y)
Z = sample_surface(X,Y)

x_sample = random.sample(set(x),100)
y_sample = random.sample(set(y),100)
z_sample = sample_surface(x_sample,y_sample) + 0.2*np.random.randn(100)

# 3D rendering of the function along with scatter
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_sample,y_sample,z_sample, c='k', depthshade=False)
ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('jet'), alpha=0.8)

# 2D contour of the mathematical function along with scatter

plt.figure()
plt.contourf(X,Y,Z, cmap=cm.get_cmap('jet'))
plt.plot(x_sample,y_sample,'ko')

# Multivariate interpolation using rbf technique applied to scatter data

rbf_interp = intp.Rbf(x_sample,y_sample,z_sample,smooth=0.2, epsilon=1)
X_int, Y_int = np.meshgrid(np.linspace(-4,4,100), np.linspace(-4,4,100)) 
Z_interp = rbf_interp(X_int,Y_int)

# Multivariate interpolation using nearest neighbour technique applied to scatter data

inputs = np.array([x_sample,y_sample]).reshape(100,2)
outputs = z_sample
ninterp = intp.NearestNDInterpolator(inputs, outputs)
Z_interp = ninterp(X_int,Y_int)

# Plot the interpolated surface

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_sample,y_sample,z_sample, c='k', depthshade=False)
ax.plot_surface(X_int,Y_int,Z_interp, cmap=cm.get_cmap('jet'), alpha=0.8)

# Plot the contour 
plt.figure()
plt.contourf(X,Y,Z, cmap=cm.get_cmap('jet'))
plt.plot(x_sample,y_sample,'ko')

plt.figure()
plt.contourf(X_int,Y_int,Z_interp, cmap=cm.get_cmap('jet'))
plt.plot(x_sample,y_sample,'ko')


