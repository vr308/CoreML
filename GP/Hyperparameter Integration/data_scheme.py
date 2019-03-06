#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:23:18 2019

@author: vidhi
"""
mean1 = [2,2]
cov1 = [[4,0],[0,1]]

mean2 = [0,0]
cov2 = [[1,0],[0,1]]

mean = [1,1]

mean_diff1 = np.subtract(mean1,mean)
mean_diff2 = np.subtract(mean2,mean)

cov_trial1 = 0.5*(np.add(cov1, cov2)) + 0.5*(np.outer(mean1, mean1) + np.outer(mean2, mean2)) - np.outer(mean, mean)
cov_trial2 = 0.5*(np.add(cov1, cov2)) + 0.5*np.outer(mean_diff1, mean_diff1) +  0.5*np.outer(mean_diff2, mean_diff2)

x_ = np.linspace(-5,5, 1000)
X1, X2 = np.meshgrid(x_,x_)
points = np.vstack([X1.ravel(), X2.ravel()]).T

mixture = lambda points : 0.5*st.multivariate_normal.pdf(points, mean1, cov1) + 0.5*st.multivariate_normal.pdf(points, mean2, cov2)
plt.contourf(X1, X2, mixture(points).reshape(1000,1000), alpha=0.2)


plt.contourf(X1, X2, st.multivariate_normal.pdf(points, mean, cov_trial1).reshape(1000,1000))
plt.contourf(X1, X2, st.multivariate_normal.pdf(points, mean, cov_trial2, allow_singular=True).reshape(1000,1000))