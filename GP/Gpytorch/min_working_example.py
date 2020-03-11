#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:07:46 2020

@author: vidhi

Minimum working example - cannot compute 'f' from 'f_rotated_'

"""

import pymc3 as pm
import theano.tensor as tt
import numpy as np
import sys
import matplotlib.pylab as plt
from pymc3.gp.util import plot_gp_dist
from pymc3.gp.util import (conditioned_vars, infer_shape,
                           stabilize, cholesky, solve_lower, solve_upper)

if __name__== "__main__":

    # number of data points
    n = 200

    # x locations
    x = np.linspace(0, 1.5, n)

    # true covariance
    l_true = 0.1
    n_true = 1.0
    cov_func = n_true**2 * pm.gp.cov.ExpQuad(1, l_true)
    K = cov_func(x[:,None]).eval()

    # zero mean function
    mean = np.zeros(n)

    # sample from the gp prior
    f_true = np.random.multivariate_normal(mean, K + 1e-6 * np.eye(n), 1).flatten()

    # link function
    def invlogit(x, eps=sys.float_info.epsilon):
        return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps

    # y-data
    y = pm.Bernoulli.dist(p=invlogit(f_true)).random()

    with pm.Model() as model:

        # hypers

        l = pm.Gamma("l", alpha=2, beta=2)
        n = 1.0
        cov = n**2 * pm.gp.cov.ExpQuad(1, l)

        gp = pm.gp.Latent(cov_func=cov)

        # make gp prior
        f = gp.prior("f", X=x[:,None], reparameterize=True)

        # logit link and Bernoulli likelihood
        p = pm.Deterministic("p", pm.math.invlogit(f))
        y_ = pm.Bernoulli("y", p=p, observed=y)

        trace = pm.sample(50, chains=1)

# Deterministic Relationship between f and f_rotated_ : f = cholesky(K).dot(f_rotated_)

    f_recovered = np.empty(shape=(50,200))

    for i in np.arange(50):
        print(i)
        l_current = trace['l'][i]
        cov_current = 1.0**2 * pm.gp.cov.ExpQuad(1, l_current)
        K = cov_current(x[:,None]) + 1e-12*np.eye(len(x))
        v = trace['f_rotated_'][i]
        f_current = cholesky(K).dot(v)
        f_recovered[i] = f_current.eval()

# f_recovered should be equal to trace['f'] but it is not..

    plt.figure()
    plt.plot(x, f_recovered.T, color='b', label='Reconstructed')
    plt.plot(x, trace['f'].T, color='grey', label='From trace')







