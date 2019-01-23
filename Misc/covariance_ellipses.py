#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:39:03 2019

@author: vidhi
"""

# Plot covariance ellipse from covariance matrix on top of the approximatied posterior

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pylab as plt

random_m = np.array([[1,-1],[2,1]])
cov_m = np.dot(random_m.T, random_m)

eig_val, eig_vec = np.linalg.eig(cov_m)
e1 =  eig_vec[:,0]
e2 = eig_vec[:,1]

# (0,0) element of the rotation matrix (think decomp of the covariance matrix) 
# is cos(theta) -> theta = cos^-1(element 0,0)

angle = 180. / np.pi * np.arccos(np.abs(eig_vec[0, 0]))

# 5.991 has been drawn from the chi-square probability table  to target a 95% confidence level for the ellipse

e = Ellipse((0,0), width=2*np.sqrt(5.991*eig_val[0]), height=2*np.sqrt(5.991*eig_val[1]), angle=angle)

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal');

# Draw an arrow from (0,0) in the direction of the eigenvector

ax.arrow(0,0, e1[0], e1[1], color='g', head_width=0.1)
ax.arrow(0,0, e1[0]*np.sqrt(5.991*eig_val[0]), e1[1]*np.sqrt(5.991*eig_val[0]), color='c')
bi_data = np.random.multivariate_normal((0,0), cov_m, size=100)

ax.scatter(bi_data[:,0], bi_data[:,1], s=1, alpha=0.2, color='b')

ax.vlines(x=0, ymin=-4, ymax=4, color='r')
ax.hlines(y=0, xmin=-6, xmax=6, color='r')
e.set_alpha(0.2)
e.set_facecolor('b')
e.set_zorder(10)
ax.add_artist(e)
ax.set_xlim(-7,7)
ax.set_ylim(-4,4)




