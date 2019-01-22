# -*- coding: utf-8 -*-
"""
Spyder Editor

In this script we consider approximating a correlated Gaussian with a factorized Gaussian. 

Parameters are known

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
np.random.seed(123)

x_ = np.linspace(-2,8,1000)

# Ground truth parameters

random_m = np.array([[1,-1],[2,1]])
pos_semi_def_m = np.dot(random_m.T, random_m)

mu_v = [2,2]
cov_m = pos_semi_def_m
precision_m = np.linalg.inv(cov_m)

# Generate data from a bi-variate Gaussian with ground truth params

np.random.multivariate_normal(mu_v, cov_m, size=10)

# Params for factorized Gaussian

m1 = mu_v[0] 
m2 = mu_v[1]

var_1 = precision_m[0][0]
var_2 = precision_m[1][1]

sns.jointplott()