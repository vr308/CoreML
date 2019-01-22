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

angle = 180. / np.pi * np.arccos(np.abs(eig_vec[0, 0]))

e = Ellipse((0,0), width=2*np.sqrt(eig_val[0]), height=2*np.sqrt(eig_val[1]), angle=angle)

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal');

e.set_alpha(0.2)
e.set_facecolor('b')
e.set_zorder(10)
ax.add_artist(e)
ax.set_xlim(-10,10)
ax.set_ylim(-6,6)