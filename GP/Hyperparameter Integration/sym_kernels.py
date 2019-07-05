#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:58:51 2019

@author: vidhi

Analytic gradients for kernels 

"""

import numpy as np 
import matplotlib.pylab as plt
import theano.tensor as tt
import pymc3 as pm
from theano.tensor.nlinalg import matrix_inverse
import pandas as pd
from sampled import sampled
from scipy.misc import derivative
import csv
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from sympy import symbols, diff, exp, log, power, sin, pi

def se_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*exp(-0.5*(1/ls**2)*(x1 - x2)**2) + noise_sd**2

def rq_kernel(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*(1 + ((x1 - x2)**2)/(2*alpha*(ls**2)))**(-alpha) + noise_sd**2

def periodic_kernel(sig_sd, p, ls, noise_sd, x1, x2):
      
      return sig_sd**2*exp(-2*sin((pi*np.abs(x1- x2)*p)/(ls**2))**2)

def matern_kernel_32(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*(1 + np.sqrt(3)(x1 - x2)/ls)*exp(-np.sqrt(3)(x1 - x2)/ls)

def matern_kernel_52(sig_sd, ls, noise_sd, x1, x2):
      
      return sig_sd**2*(1 + np.sqrt(5)(x1 - x2)/ls + 5*(x1 - x2)**2/(3*ls**2))*exp(-np.sqrt(5)(x1 - x2)/ls)