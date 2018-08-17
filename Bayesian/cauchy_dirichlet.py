# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as st 
import scipy as sp
from scipy.stats import beta
import matplotlib.pylab as plt
from statsmodels.nonparametric.kde import KDEUnivariate

c1 = st.cauchy.rvs(loc=-1,scale=2, size=1000)
c2 = st.cauchy.rvs(loc=4,scale=1.2, size=1000)

x_grid = np.linspace(-4,10, 10000)
pdf1= st.cauchy.pdf(x_grid, -1, 2)
pdf2= st.cauchy.pdf(x_grid, 4, 1.2)

plt.figure()
plt.plot(x_grid, 0.5*pdf1,linestyle='--', color='b', label='components')
plt.plot(x_grid, 0.5*pdf2, linestyle='--', color='b')
plt.plot(x_grid, 0.5*pdf1 + 0.5*pdf2, color='y', label='True dist')
plt.title('Mixture of 1d Cauchy variables')
plt.legend()

# Mixture of 2 Cauchy dist. 
c = np.ravel(np.vstack((c1,c2)))
plt.hist(c, bins=100, density=True, alpha=0.4)

kde = KDEUnivariate(c)
kde.fit(bw=1)

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

# Dirichlet dist. 

sd = np.random.dirichlet(alpha=(5,5,5), size=10000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sd[:,0],sd[:,1],sd[:,2], c='k', depthshade=False)

# Dirichlet process
# Sampling using a stick breaking view


import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import numpy as np

def dirichlet_sample_approximation(base_measure, alpha, tol=0.01):
    betas = []
    pis = []
    betas.append(beta(1, alpha).rvs())
    pis.append(betas[0])
    while sum(pis) < (1.-tol):
        s = np.sum([np.log(1 - b) for b in betas])
        new_beta = beta(1, alpha).rvs() 
        betas.append(new_beta)
        pis.append(new_beta * np.exp(s))
    pis = np.array(pis)
    thetas = np.array([base_measure() for _ in pis])
    return pis, thetas

def plot_dp_draws(alpha):
    plt.figure()
    plt.title("Dirichlet Process Sample with N(0,1) Base Measure")
    plt.suptitle("alpha: %s" % alpha)
    pis, thetas = dirichlet_sample_approximation(lambda: norm().rvs(), alpha)
    pis = pis * (norm.pdf(0) / pis.max())
    plt.vlines(thetas, 0, pis, )
    X = np.linspace(-4,4,100)
    plt.plot(X, norm.pdf(X))

plot_dp_draws(.1)
plot_dp_draws(1)
plot_dp_draws(10)
plot_dp_draws(1000)


# Draws from a DPMM

N = 5
K = 30

alpha = 2
P0 = sp.stats.norm
f = lambda x, theta: sp.stats.norm.pdf(x, theta, 0.3)


beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)





theta = P0.rvs(size=(N, K))

x_plot = np.linspace(-3, 3, 200)

dpm_pdf_components = f(x_plot[np.newaxis, np.newaxis, :], theta[..., np.newaxis])
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_plot, dpm_pdfs.T, c='gray');




# Estimating a pdf using a mixture dirichlet



