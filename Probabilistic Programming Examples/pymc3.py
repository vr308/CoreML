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
import seaborn as sns
import scipy as sp
import scipy.stats as st
from matplotlib.patches import Ellipse
import warnings 

warnings.filterwarnings('ignore')

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

# Example 5: Estimate mu and cov of a multivariate Normal

mu = pm.floatX([0., 0.])
cov = pm.floatX([[1, .5], [.5, 1.]])
y = st.multivariate_normal(mu, cov).rvs(100)

var, U = np.linalg.eig(cov)
angle = 180. / np.pi * np.arccos(np.abs(U[0, 0]))

def plot_bivariate_ellipse(y, mu, cov, new_fig):

      var, U = np.linalg.eig(cov)
      angle = 180. / np.pi * np.arccos(np.abs(U[0, 0]))
      if new_fig:
            fig, ax = plt.subplots(figsize=(8, 6))
      e = Ellipse(mu, 2 * np.sqrt(5.991 * var[0]), 2 * np.sqrt(5.991 * var[1]), angle=angle)
      e.set_alpha(0.5)
      e.set_facecolor('gray')
      e.set_zorder(10)
      #plt.add_artist(e);
      plt.scatter(y[:, 0], y[:, 1], c='k', alpha=0.5, zorder=11)
          
with pm.Model() as model:
      
    mu1 = pm.MvNormal('mu',mu=[-2,-1], cov=np.array(([1, 0],[0,1])), shape=2)
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=1, n=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    cov1 = pm.Deterministic('cov1', tt.dot(chol, chol.T))

    obs = pm.MvNormal('obs', mu=mu1, chol=chol, observed=y)

    approx_diag = pm.ADVI()
    approx_fullrank = pm.FullRankADVI()
     
    approx_diag.fit()
    approx_fullrank.fit()
    
    trace1 = approx_diag.approx.sample()
    trace2 = approx_fullrank.approx.sample()
    #trace_nuts = pm.sample()
    
cov1 = []
cov2 = []
for i in np.arange(500):
    cov1.append(trace1['cov1'][i][1][0])
    cov2.append(trace2['cov1'][i][1][0])
    
sns.kdeplot(trace_nuts['mu'][:,0], label='HMC', color='b', shade=True, alpha=0.2)
sns.kdeplot(trace_nuts['mu'][:,1], color='b',shade=True, alpha=0.2 )

sns.kdeplot(trace1['mu'][:,0], label='VI (MF)', color='g', shade=True, alpha=0.2)
sns.kdeplot(trace1['mu'][:,1], color='g', shade=True, alpha=0.2)

sns.kdeplot(trace2['mu'][:,0], label='VI (FR)', color='r', shade=True, alpha=0.2)
sns.kdeplot(trace2['mu'][:,1], color='r', shade=True, alpha=0.2)

# Extract the converged mu and cov from approx

mu_nuts = trace_nuts['mu'].mean(axis=0)
mu_diag = trace1['mu'].mean(axis=0)
mu_full = trace2['mu'].mean(axis=0)

cov_nuts = trace_nuts['cov1'].mean(axis=0)
cov_diag = trace1['cov1'].mean(axis=0)
cov_full = trace2['cov1'].mean(axis=0)

plot_bivariate_ellipse(y, mu_nuts, cov_nuts, True)
plot_bivariate_ellipse(y, mu_diag, cov_diag, True)
plot_bivariate_ellipse(y, mu_full, cov_full, True)

n_dim = 2
data = np.random.randn(10000, n_dim)

with pm.Model() as model:
    # Note that we access the distribution for the standard
    # deviations, and do not create a new random variable.
    mu1 = pm.MvNormal('mu',mu=[-2,-1], cov=np.array(([1, 0],[0,1])), shape=2)
    sd_dist = pm.HalfCauchy.dist(beta=2.5)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=n_dim, 
                                    sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(n_dim, packed_chol, lower=True)
    cov = pm.Deterministic('cov', tt.dot(chol, chol.T))

    # Define a new MvNormal with the given covariance
    vals = pm.MvNormal('vals', mu=mu1, 
                       cov=cov, shape=n_dim,
                       observed=y)
    
    advi = pm.ADVI()
    advi.fit()
    trace_advi = advi.approx.sample()
    
y = np.random.normal(1,3, 10000)

with pm.Model() as model:
      
      mu = pm.Uniform('mu', -10, 10)
      #sigma = pm.Uniform('sigma', 0, 10)
      sigma = 3
      #mu = pm.Normal('mu', 0, 10)
      #sigma = pm.Gamma('sigma', 2, 2)
      obs = pm.Normal('obs', mu=mu, sd=sigma, observed=y)      
      advi = pm.ADVI(n=50000)
      tracker = pm.callbacks.Tracker(
      mean=advi.approx.mean.eval, # callable that returns mean
      )
      advi.fit(callbacks=[tracker])
      trace_advi = advi.approx.sample(10000)
      

trace_mean = pm.summary(trace_advi)['mean']['mu']
trace_std = pm.summary(trace_advi)['sd']['mu']
  
means = advi.approx.bij.rmap(advi.approx.mean.eval())  
std = advi.approx.bij.rmap(advi.approx.std.eval())  

# These should roughly match trace

approx_mean = model.mu_interval__.distribution.transform_used.backward(means['mu_interval__']).eval()
approx_std = model.mu_interval__.distribution.transform_used.backward(std['mu_interval__']).eval()

forward_eps = lambda x: model.mu_interval__.distribution.transform_used.forward_val(x)
backward_theta = lambda x: model.mu_interval__.distribution.transform_used.backward(x).eval()
j_factor = lambda x: np.exp(model.mu_interval__.distribution.transform_used.jacobian_det(x).eval())

x = np.linspace(-2, 5, 1000)

plt.figure()
plt.hist(trace_advi['mu'], bins=100, normed=True)
plt.plot(x, st.norm.pdf(forward_eps(x), means['mu_interval__'], std['mu_interval__'])*j_factor(forward_eps(x)))
density = lambda x : st.norm.pdf(forward_eps(x), approx_mean, approx_std)*j_factor(backward_theta(x))

sp.integrate.quad(density, -2, +5)

# This matches

plt.figure()
plt.plot(x, st.norm.pdf(x, means['mu_interval__'], std['mu_interval__']))
plt.plot(x, st.norm.pdf(backward_theta(x), approx_mean, approx_std)*j_factor(x))


fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
mu_ax.plot(np.array(tracker['mean'])[:,0])
mu_ax.set_title('Mean track')
std_ax.plot(np.exp(np.array(tracker['mean'])[:,1]))
std_ax.set_title('Std track')
hist_ax.plot(advi.hist)
hist_ax.set_title('Negative ELBO track');
