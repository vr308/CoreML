#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:33:04 2017

@author: vr308
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-1, 7, N)
Y = np.linspace(-1, 7, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([4, 4])
Sigma = np.array([[ 1. , 0.5], [0.5,  1.0]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

fig = plt.figure()
ax = fig.gca(projection='3d')
surface = ax.plot3D(xs=[X],ys=[Y],zs=[Z])
surface.set_data([X,Y])
 
def init():
    surface.set_date([X],[Y],[])
    return surface,
    
def animate(i):
    #X = np.linspace(-3,3,N)
    #Y = np.linspace(-3,4,N)
    #Z = multivariate_gaussian(pos, mu, Sigma)[0:i]
    Z1 = Z[0:i]
    surface.set_data([X,Y])
    surface.set_3d_properties(zs=Z1)
    return surface,
    
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, blit=True)
plt.show()
    
    
    
    
    
    
    
    
# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)


# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.Reds)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.Reds)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)








