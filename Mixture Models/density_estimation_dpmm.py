#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vidhi

# Sample posterior densities from estimated mixtures - in the sklearn framework 
# Construct a density from a histogram and sample from it
# Fit DPGMM and BGM to the cms data.

"""

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import pandas as pd
import pymc3 as pm
import scipy 

N=100000

# Chris data 

home_path = '~/Desktop/Workspace/CoreML/Mixture Models/'
uni_path = '/home/vidhi/Desktop/Workspace/CoreML/Mixture Models/'
      
path = uni_path

data = pd.read_csv(path + 'Data/1dDensity.csv', sep=',', names=['x','density'])
data['prob'] = data['density']/ np.sum(data['density'])
data['cdf'] = np.cumsum(data['prob'])

log_data = np.log(data)
log_data['prob'] = log_data['density']/np.sum(log_data['density'])
log_data['cdf'] = np.cumsum(log_data['prob'])


plt.figure()
plt.plot(data['x'], data['prob'], '-')
plt.plot(data['x'], data['density'], '-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log_x')
plt.ylabel('log_density')
plt.title('Raw data')

plt.figure()
plt.plot(np.log(data['x']), np.log(data['density']))
plt.xlabel('log(x)')
plt.ylabel('log(density)')

u = np.random.uniform(np.min(log_data['cdf']), 1, (N, ))
u = np.random.uniform(np.min(data['cdf']), 1, (N, ))

#Find the location 
   
samples = []
for i in u:
    loc = np.where(log_data['cdf'] <= i)[0][-1]
    lower_limit = log_data['x'][loc]
    upper_limit = log_data['x'][loc+1]
    samples.append(np.random.uniform(lower_limit, upper_limit))


plt.figure()
plt.hist(samples, bins=500, density=True, alpha=0.8)
plt.title('Samples generated using inverse CDF', fontsize='small')
plt.xscale('log')
plt.xlim(0,100)
plt.ylim(0,16)

# Given 2 points of a discrete distribution, form a continuous pdf connecting the two points

point_A = (log_data['x'][19], log_data['density'][19])
point_B = (log_data['x'][30], log_data['density'][30])

x = [point_A[0], point_B[0]]
h = [point_A[1], point_B[1]]

def slope_(point_A, point_B):
      
      x1 = point_A[0]
      h1 = point_A[1]
      x2 = point_B[0]
      h2 = point_B[1]
      slope = (h2 - h1)/(x2 - x1)
      return slope

slope = slope_(point_A, point_B) 

normalizer = 0.5*(point_A[1] + point_B[1])*(point_B[0] - point_A[0])

func = lambda x : x*slope + point_A[1] - slope*point_A[0]
pdf = lambda x : (x*slope + point_A[1] - slope*point_A[0])/normalizer

# Check if the pdf is valid 
scipy.integrate.quad(pdf, x[0], x[1])

cdf = lambda x : 1/normalizer * (h[0]*(x - x[0]) + slope*(x**2/2 - x*x[0] + x[0]**2/2))
cdf_inv = lambda x : np.sqrt((x*normalizer + h[0]**2/(2*slope))/(slope/2)) - h[0]/slope + x[0]

x_range = np.arange(point_A[0],point_B[0],0.01)
y_range = func(x_range)
y_pdf = pdf(x_range)
y_cdf = cdf(x_range)

plt.figure()
plt.plot(point_A[0], point_A[1], 'bo')
plt.plot(point_B[0], point_B[1], 'bo')
plt.plot(x, h, 'r-')
plt.plot(x_range, y_range)
plt.plot(x_range, y_pdf)
plt.plot(x_range, y_cdf)


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

K = 11

with pm.Model() as model:
    
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 5, alpha, shape=K)
    
    w = pm.Deterministic('w', stick_breaking(beta))
    
    mu = pm.Uniform('mu', -2., 8., shape=K, testval=start['mu'])
    sd = pm.Uniform('sd', 0.001, 3, shape=K, testval=start['sd'])
    
    obs = pm.Mixture('obs', w, pm.Normal.dist(mu, sd), observed=np.array(samples[::10]))
    
    
 start = {'mu': np.array([-0.2, 0, 1.2, 2.3, 4.5, -0.5, 0, 1,2,3,4]).reshape(11,), 
          'sd': np.array([0.03, 0.03, 0.045, 0.03, 0.03, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(11,),
          'beta': np.array([0.0206186, 0.0421053, 0.10989, 0.0246914, 0.0506329, 0.0666667, 0.257143, 0.307692, 0.388889, 0.545455, 1.]).reshape(11,)}

with model:
    trace = pm.sample(500, discard_tuned_samples=False, chains=1)





x_plot = np.linspace(-2, 8,1000)
post_pdf_contribs = st.norm.pdf(np.atleast_3d(x_plot),
                                         trace['mu'][:, np.newaxis, :], trace['sd'][:,np.newaxis,:])
post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)

plt.figure()
plt.plot(x_plot, post_pdfs.T, c='gray');
plt.hist(samples[::10], bins=100, density=True)
plt.plot(x_plot, post_pdfs.mean(axis=0),
        c='k', label='Posterior expected density');
plt.fill_between(x_plot, post_pdf_low, post_pdf_high,
                 color='red', alpha=0.45)
plt.title('Non-parametric (loc-scale) mixture of Gaussians')
plt.ylim(0,0.7)


plt.figure()
plt.plot(x_plot, post_pdf_contribs[0], color='b')
plt.plot(x_plot, post_pdf_contribs[1], color='g')

