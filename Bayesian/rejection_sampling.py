#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:36:15 2018

@author: vr308
"""

### Rejection sampling on a continuous distribution 

from scipy.stats import norm, uniform
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The multiplication constant to make our probability estimation fit
M = 3
# Number of samples to draw from the probability estimation function
N = 1000

# The target probability density function
f = lambda x: 0.6 * norm.pdf(x, 0.35, 0.05) + 0.4 * norm.pdf(x, 0.65, 0.08)

# The approximated probability density function
g = lambda x: norm.pdf(x, 0.45, 0.2)

# A number of samples, drawn from the approximated probability density function
x_samples = M * np.random.normal(0.45, 0.2, (N,))

# A number of samples in the interval [0, 1]
u = np.random.uniform(0, 1, (N, ))

# Now examine all the samples and only use the samples found by rejection sampling
samples = [(x_samples[i], u[i] * M * g(x_samples[i])) for i in range(N) if u[i] < f(x_samples[i]) / (M * g(x_samples[i]))]

# Make the plots
fig, ax = plt.subplots(1, 1)

# The x coordinates
x = np.linspace(0, 1, 500)

# The target probability function
ax.plot(x, f(x), 'r-', label='$f(x)$')

# The approximated probability density function
ax.plot(x, M * g(x), 'b-', label='$M \cdot g(x)$')

# The samples found by rejection sampling
ax.plot([sample[0] for sample in samples], [sample[1] for sample in samples], 'g.', label='Samples')

# Set the window size
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 6])

# Show the legend
plt.legend()

# Set the title
plt.title('Rejection sampling')

# Show the plots
plt.show()

### Rejection sampling on a discrete distribution 

N = 30000

x = [1,2,3,4,5]
p = [0.2,0.1,0.1,0.3,0.3]

q = stats.rv_discrete(name='q', values=(x,[1.0/len(x)]*5))
M = 1.8

plt.figure()
plt.plot(x,p,'r-')
plt.bar(x, p, color='g')
plt.plot(x, M*q.pmf(x),'b-')

x_samples = np.random.randint(1, 6, (N,))

# A number of samples in the interval [0, 1]
u = np.random.uniform(0, 1, (N, ))

samples = pd.Series([(x_samples[i]) for i in range(N) if u[i] < p[x_samples[i]-1] / (M * q.pmf(x_samples[i]))])

# Display the samples and compare to the target distribution
sumt = len(samples)
counts = [samples.value_counts()[x]/sumt for x in [1,2,3,4,5]]

plt.figure()
plt.bar(x,counts)
plt.plot(x,p,'r-')



















