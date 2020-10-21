#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

import numpy as np
import pymc3 as pm 
import seaborn as sns

X = np.linspace(-10, 10, 100)[:,None]
y = (X**3 - X**2 + X)

lengthscale = 2
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

K = cov(X).eval()
plt.matshow(K)

def warp_func(x, a, b, c):
    return 1.0 + x + (a * tt.tanh(b * (x - c)))


def logistic(x, a, x0, c, d):
    # a is the slope, x0 is the location
    return d * pm.math.invlogit(a*(x - x0)) + c
c = 0.1
d = 2.0

a = 2.0
import theano.tensor as tt
import matplotlib.pylab as plt
x0 = 5.0
cov1 = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(-a, x0, c, d),
                           cov_func=pm.gp.cov.ExpQuad(1, 0.2))
cov2 = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a, x0, c, d),
                           cov_func=pm.gp.cov.Cosine(1, 0.5))
cov = cov1 + cov2
K = cov(X).eval()

plt.matshow(K)

plt.figure(figsize=(14,4))
plt.fill_between(X.flatten(), np.zeros(100), logistic(X.flatten(), -a, x0, c, d).eval(), label="ExpQuad region",
                 color="slateblue", alpha=0.4);
plt.fill_between(X.flatten(), np.zeros(100), logistic(X.flatten(), a, x0, c, d).eval(), label="Cosine region",
                 color="firebrick", alpha=0.4);
plt.legend();
plt.xlabel("X"); plt.ylabel("$\phi(x)$");
plt.title("The two scaling functions");

plt.figure(figsize=(14,4))
plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=3).T);
plt.title("Samples from the GP prior");
plt.ylabel("y");
plt.xlabel("X");

X_gaussian = pm.MvNormal.dist(mu=np.zeros(2), cov=np.eye(2)).random(size=100)
sns.kdeplot(X_gaussian, shade=True, bw=2)

lengthscale = [10,0.1]
eta = 2.0
cov_se = eta**2 * pm.gp.cov.ExpQuad(2, lengthscale)
cov_per = eta**2 * pm.gp.cov.Periodic(2, period=1, ls=lengthscale)

K_rbf = cov_se(X_gaussian).eval()
K_per = cov(X_gaussian).eval()

K = K_per

warping_func1 = pm.MvNormal.dist(mu=np.zeros(K_per.shape[0]), cov=K_per).random(size=1).T
warping_func2 = pm.MvNormal.dist(mu=np.zeros(K_per.shape[0]), cov=K_per).random(size=1).T

sns.kdeplot(np.vstack((warping_func1, warping_func2)).T, shade=2)
plt.plot(warping_func1, warping_func2, 'ro')

# 

