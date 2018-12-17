#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vidhi
"""

import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import pymc3 as pm
plt.style.use("ggplot")


n = 50
z = 10
alpha = 12
beta = 12
alpha_post = 22
beta_post = 52

# How many iterations of the Metropolis 
# algorithm to carry out for MCMC
iterations = 100

# Use PyMC3 to construct a model context
basic_model = pm.Model()
with basic_model:
    # Define our prior belief about the fairness
    # of the coin using a Beta distribution
    theta = pm.Beta("theta", alpha=alpha, beta=beta)

    # Define the Bernoulli likelihood function
    y = pm.Binomial("y", n=n, p=theta, observed=z)

    # Use the Metropolis algorithm (as opposed to NUTS or HMC, etc.)
    step = pm.Metropolis()

    # Calculate the trace
    trace = pm.sample(iterations, step, progressbar=True)
    

# Plotting