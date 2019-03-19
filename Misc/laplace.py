#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:25:32 2019

@author: vidhi

Laplace approximation for arbitrary distributions in 1d and 2d

"""
import pymc3 as pm
import scipy.stats as st
import tensorflow as tf

import six
import edward as ed
from edward.inferences.map import MAP
from edward.models import PointMass, RandomVariable
from edward.util import get_session, get_variables
from edward.util import copy, transform
import matplotlib.pylab as plt

try:
  from edward.models import \
      MultivariateNormalDiag, MultivariateNormalTriL, Normal, Gamma
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))

# Laplace in pymc3 for arbitary distribution - doesn't work in the expected way

with pm.Model() as model:
      
    g = pm.Gamma('g', 8, 2)
    #h = pm.Normal('n', 2,1)
    #h = pm.Beta('h', 2,2)
    map_estimate = pm.find_MAP(vars=[g])
    hessian = pm.find_hessian(map_estimate)
      
hessian_ = (g.distribution.alpha.eval() - 1)/(map_estimate['g']**2)
sd = np.sqrt(1/hessian_)

x_ = np.linspace(0,20, 1000)
plt.plot(x_, st.gamma.pdf(x_, g.distribution.alpha.eval(), scale=1/g.distribution.beta.eval()))
plt.plot(x_, st.norm.pdf(x_, map_estimate['g'],sd))
      
# Laplace in edward for arbitary distribution 

D = 1
w = Gamma(concentration=[2.0], rate=[2.0])
mu = tf.Variable(tf.random_normal([1]))
sd = tf.Variable(tf.random_normal([1]))

qw = Normal(loc=mu, scale=sd)

inference = ed.Laplace({w: qw}, data=None)
inference.run()

sess=ed.get_session()
mu, sd = sess.run([mu,sd])

x_ = np.linspace(0,20, 1000)
plt.plot(x_, st.gamma.pdf(x_, 2, scale=1/w.rate.eval()))
plt.plot(x_, st.norm.pdf(x_, mu, sd))