# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as st 
import sklearn as skt
from scipy.stats import beta
from scipy.stats import dirichlet
import matplotlib.pylab as plt
from statsmodels.nonparametric.kde import KDEUnivariate
from mpl_toolkits.mplot3d import Axes3D

c1 = st.cauchy.rvs(loc=-1,scale=0.2, size=10)
c2 = st.cauchy.rvs(loc=0.8,scale=0.2, size=10)

plt.figure()
plt.hist(c1, bins=50)
plt.hist(c2, bins=50)

# Mixture of 2 Cauchy dist. 
c = np.ravel(np.vstack((c1,c2)))
plt.hist(c, bins=50, density=True)

kde = KDEUnivariate(c)
kde.fit(bw=0.2)

x_grid = np.linspace(-4,10, 10000)
plt.plot(x_grid, kde.evaluate(x_grid))

plt.title('Mixture of Cauchy\'s and KDE estimate')

# Beta distribution 

samples_beta = np.random.beta(7,10, size=10000)

x = np.linspace(0,1,1000)

plt.figure()
plt.hist(samples_beta, bins=100)

alpha=[0.1,0.5,0.7]
beta_=alpha

ab = [(0.5,0.5), (5,1), (1,2), (1,5), (2,4),(1,1)]

means=[]
for i in ab:
    means.append(i[0]/(i[0] + i[1]))
    
for i in ab:
        plt.plot(x,beta.pdf(x, i[0], i[1]), label='(' + str(i[0]) + ',' +  str(i[1]) + ')')
        plt.vlines(x=(i[0]/(i[0] + i[1])), ymin=0, ymax=3,label='Mean')
plt.legend()

#Bayesian updates with beta distribution and coin toss

prior_beta = beta.pdf(x,2,2)

N = 13
m = 9

posterior_beta = beta.pdf(x, 11,6)

plt.figure()
plt.plot(x, prior_beta)
plt.plot(x, posterior_beta)

# Dirichlet
sd = np.random.dirichlet(alpha=(5,5,5), size=10000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sd[:,0],sd[:,1],sd[:,2], c='k', depthshade=False)


