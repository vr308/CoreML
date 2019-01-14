#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:34:42 2019

@author: vidhi
"""

import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.patches import Ellipse
import seaborn as sns
import scipy as sp

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

# Example 0: Beta-Binomial model

x_beta_binomial = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

with pm.Model() as bb_model:
      
      p = pm.Uniform('p', 0, 1) # prior
      x_obs = pm.Bernoulli('y', p, observed=x_beta_binomial)
      trace = pm.sample(5000)
      
pm.traceplot(trace, combined=True)

# plot the true beta-binomial posterior distribution
fig, ax = plt.subplots()

prior = sp.stats.uniform(0, 1)
posterior = sp.stats.beta(1 + x_beta_binomial.sum(), 1 + (1 - x_beta_binomial).sum())

plot_x = np.linspace(0, 1, 100)
ax.plot(plot_x, prior.pdf(plot_x),
        '--', c='k', label='Prior');
ax.plot(plot_x, posterior.pdf(plot_x),
        c='b', label='Posterior');
ax.set_xticks(np.linspace(0, 1, 5));
ax.set_xlabel(r'$p$');
ax.set_yticklabels([]);
ax.legend(loc=1);
ax.hist(trace['p'], bins=50, normed=True,
        color='g', lw=0., alpha=0.5,
        label='MCMC approximate posterior');

ax.legend();

# Example 1 : Prints the log-likelihood of the model 

data = np.random.randn(100)

with pm.Model() as model:
       
       mu = pm.Normal('mu', mu=0, sd=1)
       obs = pm.Normal('obs', mu=mu, sd=1, observed=data)
       print(model.logp({'mu':0}))
       

# Example 2: Do inference, where we are trying to predict the mean of the normal distribution
       
data = np.random.normal(1,1,100)

with pm.Model() as model:
      
      mu = pm.Normal('mu', mu=0, sd=1) # prior distribution
      obs = pm.Normal('obs', mu=mu, sd=1, observed=data)
      trace = pm.sample(1000, tune=500 ) #discard_tuned_samples=False
      
d=pm.traceplot(trace, combined=True, priors=[pm.Normal.dist(0,1)])
pm.plot_posterior(trace)

# Example 3: VI simple example

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    sd = pm.HalfNormal('sd', sd=1)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=np.random.randn(100))
    approx_ADVI = pm.fit()
    approx_fullrankADVI = pm.fit(method='fullrank_advi')


trace_advi = approx_ADVI.sample(500)
trace_fullrankADVI = approx_fullrankADVI.sample(500)

pm.traceplot(trace_advi)
pm.traceplot(trace_fullrankADVI)

# Example 4: MCMC animation

x_animation = np.linspace(0, 1, 100)
y_animation = 1 - 2 * x_animation + np.random.normal(0., 0.25, size=100)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_animation, y_animation,
           c='b');
ax.set_title('MCMC Animation Data Set');

with pm.Model() as mcmc_animation_model:
    intercept = pm.Normal('intercept', 0., 10.)
    slope = pm.Normal('slope', 0., 10.)
    tau = pm.Gamma('tau', 1., 1.)
    y_obs = pm.Normal('y_obs', intercept + slope * x_animation, tau=tau,
                      observed=y_animation)
    animation_trace = pm.sample(5000)


animation_cov = np.cov(animation_trace['intercept'],
                       animation_trace['slope'])

animation_sigma, animation_U = np.linalg.eig(animation_cov)
animation_angle = 180. / np.pi * np.arccos(np.abs(animation_U[0, 0]))

animation_fig = plt.figure()

e = Ellipse((animation_trace['intercept'].mean(), animation_trace['slope'].mean()),
            2 * np.sqrt(5.991 * animation_sigma[0]), 2 * np.sqrt(5.991 * animation_sigma[1]),
            angle=animation_angle, zorder=5)
e.set_alpha(0.5)
e.set_facecolor('b')
e.set_zorder(9);

animation_images = [(plt.plot(animation_trace['intercept'][-(iter_ + 1):],
                              animation_trace['slope'][-(iter_ + 1):],
                              '-o', c='k', alpha=0.5, zorder=10)[0],)
                    for iter_ in range(50)]

animation_ax = animation_fig.gca()
animation_ax.add_artist(e);

animation_ax.set_xticklabels([]);
animation_ax.set_xlim(0.75, 1.3);

animation_ax.set_yticklabels([]);
animation_ax.set_ylim(-2.5, -1.5);

mcmc_animation = ArtistAnimation(animation_fig, animation_images,
                                 interval=100, repeat_delay=5000,
                                 blit=True)


# Example 5: VI example multivariate rv

mu = pm.floatX([0., 0.])
cov = pm.floatX([[1, .5], [.5, 1.]])
with pm.Model() as model:
    pm.MvNormal('x', mu=mu, cov=cov, shape=2)
    approx = pm.fit(method='fullrank_advi')
    trace = approx.sample(200)

sns.jointplot(trace['x'][:,0], trace['x'][:,1], kind='kde')

# Figuring out how to plot covariance ellipse from covariance matrix on top of the approximatied posterior

eig_val, eig_vec = np.linalg.eig(cov)
e1 =  eig_vec[:,0]
e2 = eig_vec[:,1]

angle = 180. / np.pi * np.arccos(np.abs(eig_vec[0, 0]))

e = Ellipse((0,0), width=2*np.sqrt(eig_val[0]), height=2*np.sqrt(eig_val[1]), angle=angle)

fig = plt.figure()
ax = fig.gca()

e.set_alpha(0.2)
ax.add_artist(e)
ax.set_xlim(-2,2)
ax.set_ylim(-1,1)