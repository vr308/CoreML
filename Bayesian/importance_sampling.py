# -*- coding: utf-8 -*-
"""
Spyder Editor

Importance sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
np.random.seed(123)

x_ = np.linspace(-2,8,1000)

target_mu = 3
target_sd = 1

proposal_mu = 3
proposal_sd = (1/np.sqrt(2))*target_sd
proposal_sd = 1.3*target_sd

target = lambda x : st.norm.pdf(x,target_mu, target_sd)
proposal = lambda x:  st.norm.pdf(x, proposal_mu, proposal_sd)

def importance_sampling(target, proposal, proposal_mu, proposal_sd):
    
    samples = []
    x = st.norm.rvs(proposal_mu, proposal_sd, 10000)
    weights = target(x)/proposal(x)
    for i in np.arange(10000):
          k = np.min((1, weights[i]))
          u = st.uniform(0,1).rvs()
          if u < k:
                samples.append(x[i])
          else:
                samples.append(x[i-1])
    return samples
    

samples = importance_sampling(target, proposal, proposal_mu, proposal_sd)

#norm_weights = weights/np.sum(weights)
#samples = np.random.choice(x,size=10000, replace=True, p=norm_weights)

plt.figure()
plt.plot(x_, target(x_), label='target')
plt.plot(x_, proposal(x_), label='proposal')
plt.hist(samples, density=True, bins=100)
plt.legend()

