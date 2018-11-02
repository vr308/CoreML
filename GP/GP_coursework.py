#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:44:45 2018

@author:  vr308
"""

import scipy.io
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as Ck, WhiteKernel, CompoundKernel, ExpSineSquared, Product
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib.pylab as plt
import itertools as it
from matplotlib.colors import LogNorm
import numpy as np

data = scipy.io.loadmat('cw1a.mat')
x_train = data['x']
y_train = data['y']
x_test = np.linspace(-3,3,200)

# Instansiate a Gaussian Process model
#kernel = Ck(1000.0, (1e-10, 1e3)) * RBF(2, (1, 100)) + WhiteKernel(0.1, noise_level_bounds =(1e-5, 1e2))
kernel = Ck(10.0, (1e-10, 1e3)) * RBF(3, length_scale_bounds=(0.5, 3)) + WhiteKernel(10, noise_level_bounds=(1e-5,50))
kernel_se = Ck(10.0, (1e-10, 1e3)) * RBF(3, length_scale_bounds=(0.5, 3)) + WhiteKernel(10, noise_level_bounds=(1e-5,50))
kernel_per = ExpSineSquared(1,1)
kernel = Product(kernel_se, kernel_per)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gpr.fit(x_train,y_train)
y_mean_pos, y_std_pos = gpr.predict(x_test.reshape(-1,1), return_std=True) 

#rmse_ = np.round(np.sqrt(np.mean(np.square(y_pred_test - y_test))),2)
lml = np.round(gpr.log_marginal_likelihood_value_,2)

plt.figure()
#plt.plot(x_test, f(x_test), 'r:', label=u'$f(x) = x^2\sin(x)$')
plt.plot(x_train, y_train, 'k.', markersize=8, label=u'Observations')
plt.plot(x_test, y_mean_pos, 'b-', label=u'Prediction')
plt.fill_between(np.ravel(x_test), np.ravel(y_mean_pos) - 1.96*y_std_pos, np.ravel(y_mean_pos) + 1.96*y_std_pos, alpha=0.2, color='k')
plt.title('GPR ' +'\n' + str(gpr.kernel_) + '\n' + 'Min value of log marginal likelihood: ' +  str(lml), fontsize='small')
plt.legend(fontsize='small')

# LML surface

lengthscale = np.logspace(-2,3,50)
noise_variance = np.logspace(-4,2,50)
l, n = np.meshgrid(lengthscale, noise_variance)
ln = np.array(list(it.product(lengthscale,noise_variance)))
lml_surface = []
for i in range(2500):
    lml_surface.append(gpr.log_marginal_likelihood(([np.log(248*248), np.log(ln[i][0]), np.log(ln[i][1])])))

lml = np.array(lml_surface).reshape(50,50).T
vmin, vmax = (-lml).min(), (-lml).max() 
vmax = 10000
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 2000), decimals=1)

plt.figure()
plt.contourf(l,n, -lml, levels=level, cmap=cm.get_cmap('jet'),norm=LogNorm(vmin=vmin, vmax=vmax))
plt.plot(np.exp(gpr.kernel_.theta[1]), np.exp(gpr.kernel_.theta[2]), 'r+', label='LML Local Minimum')
plt.colorbar(format='%.1f')
plt.xscale("log")
plt.yscale("log")
plt.legend()