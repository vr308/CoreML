#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:29:57 2018

"""

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# Concentration of measure

plt.figure()
for i in [2,10,25]:
      samples = np.random.multivariate_normal([0]*i, np.eye(i), 1000)
      dist = np.linalg.norm(samples, axis=1)
      avg_dist = dist/np.sqrt(i)     
      #print(np.mean(dist))
      print(np.var(avg_dist))
      #plt.hist(avg_dist, bins=100)
      sns.kdeplot(avg_dist, kernel='gau')

# You can either divide by the sqrt(dimension) or the mean, it is the same thing. 
      