#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

import torch
import pyro.distributions as dist
from torch import nn
import torch.tensor as T
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
import matplotlib.pyplot as plt
import numpy as np
import pyro.distributions.transforms as trans
from sklearn import datasets
import pyro
import seaborn as sns
import matplotlib.patches as mpatches

# Fitting a diagonal Gaussian distribution to a non-diagonal one
# Generate some data

mean = torch.tensor([4.0, 3.0])
covariance_matrix  = torch.tensor([[2.0,1.0],[1.0,2.0]])

X = dist.MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix).rsample(sample_shape=torch.Size([1000]))
X = torch.tensor(X, dtype=torch.float)

############
#
# Using pure pyro: model() and guide() framework
#
# Fitting a Gaussian distribution using SVI
# Saving the trace of params
#
############

# define the model .i.e how the data was generated, usually entails prior and likelihood
def model(X):
    mu = pyro.sample('mu', dist.Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])))
    sd = pyro.sample('sd', dist.LogNormal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])))
    for i in range(len(X)):
        pyro.sample('obs_{}'.format(i), dist.Normal(mu, sd), obs=X[i])
    
# define the guide / variational family

def guide(X):
    mu_loc = pyro.param('mu_loc', torch.tensor([0.0,0.0]))
    mu_sd = pyro.param('mu_sd', torch.tensor([1.0,1.0]), constraint = constraints.positive)
    sd_loc = pyro.param('sd_loc', torch.tensor([0.0,0.0]))
    sd_sd = pyro.param('sd_sd', torch.tensor([1.0,1.0]), constraint = constraints.positive)
    
    mu = pyro.sample('mu', dist.Normal(mu_loc, mu_sd))
    sd = pyro.sample('sd', dist.LogNormal(sd_loc, sd_sd))
    #mu = pyro.sample('mu', dist.Delta(mu_loc))
    #sd = pyro.sample('sd', dist.Delta(sd_loc))
    #pyro.sample('')

adam_params = {"lr": 0.05}
optimiser = pyro.optim.Adam(adam_params)
# setup the inference algorithm
svi = pyro.infer.SVI(model, guide, optimiser, loss=Trace_ELBO())

# do gradient steps
losses, mu_loc, mu_sd, sd_loc, sd_sd = [],[],[],[],[]
for _ in range(700):
    losses.append(svi.step(X))
    mu_loc.append(pyro.param('mu_loc').clone().detach().numpy())
    mu_sd.append(pyro.param('mu_sd').clone().detach().numpy())
    sd_loc.append(pyro.param('sd_loc').clone().detach().numpy())
    sd_sd.append(pyro.param('sd_sd').clone().detach().numpy())
    print(pyro.param('mu_loc'))
    #print(pyro.param('sd_loc'))
    if _ % 10 == 0:
        print('.', end='')

pyro.clear_param_store()

#samples from the trained variational distribution
X_learnt = dist.MultivariateNormal(loc=mu_loc.detach(), covariance_matrix=torch.eye(2)*sd_loc.detach()).rsample(sample_shape=torch.Size([1000]))
        
fig = plt.figure(figsize=(10,4)) 
axes = fig.subplots(1,3)
ax1 = sns.kdeplot(X[:,0], X[:,1],bw_method=5, shade=True, color='b', label='True', ax=axes[0])
ax2 = sns.kdeplot(X_learnt[:,0], X_learnt[:,1], bw_method=5, color='r', shade=True, label='Learnt',  ax=axes[0], legend=False)
axes[0].set_title('Learning a Gaussian')
handles = [mpatches.Patch(facecolor='b', label="True"),
           mpatches.Patch(facecolor='r', label="Learnt")]
axes[0].legend(handles, ['True', 'Learnt'])
plt.subplot(132)
plt.plot(mu_loc, label='loc trace')
plt.fill_between(range(len(mu_loc)), np.array(mu_loc)[:,0]-2*np.array(mu_sd)[:,0], np.array(mu_loc)[:,0] + 2*np.array(mu_sd)[:,0], alpha=0.4)
plt.fill_between(range(len(mu_loc)), np.array(mu_loc)[:,1]-2*np.array(mu_sd)[:,1], np.array(mu_loc)[:,1] + 2*np.array(mu_sd)[:,1], alpha=0.4)
plt.legend() 

plt.subplot(133)
plt.plot(sd_loc, label='sd trace')
plt.fill_between(range(len(sd_loc)), np.array(sd_loc)[:,0]-2*np.array(sd_sd)[:,0], np.array(sd_loc)[:,0] + 2*np.array(sd_sd)[:,0], alpha=0.4)
plt.fill_between(range(len(sd_loc)), np.array(sd_loc)[:,1]-2*np.array(sd_sd)[:,1], np.array(sd_loc)[:,1] + 2*np.array(sd_sd)[:,1], alpha=0.4)
plt.legend() 
      
############
#
# Using torch instead of pyro
#
# Fitting a Gaussian distribution using MLE
# Saving the trace of parama
#
############

mu = torch.nn.Parameter(torch.tensor([0.0,0.0]))
sd = torch.nn.Parameter(torch.tensor([1.0,1.0]))

#mu = pyro.param('mu', torch.tensor([0.0,0.0]))
#sd = pyro.param('sd', torch.tensor([1.0,1.0]))
base_dist = dist.Normal(mu, sd)
#spline_transform = T.Spline(2, count_bins=16)
#flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

losses, mu_trace, sd_trace = [],[],[]
D = torch.tensor(X, dtype=torch.float)
optimiser = torch.optim.Adam([mu,sd], lr=0.01)
for step in range(2000):
    optimiser.zero_grad()
    loss = -base_dist.log_prob(D).mean()
    loss.backward()
    optimiser.step()
    losses.append(loss) 
    #print(mu_trace)
    mu_trace.append(mu.clone().detach().numpy())
    sd_trace.append(sd.clone().detach().numpy())
    if step % 50 == 0:
        print('step: {}, loss: {}, mu: {}, sd: {}'.format(step, loss.item(), mu.detach(), sd.detach()))

X_learnt = dist.MultivariateNormal(loc=mu.detach(), covariance_matrix=torch.eye(2)*sd.detach()).rsample(sample_shape=torch.Size([1000]))

# Plotting
fig = plt.figure(figsize=(8,4))
axes = fig.subplots(1,2)
ax1 = sns.kdeplot(X[:,0], X[:,1],bw_method=5, shade=True, color='b', label='True', ax=axes[0])
ax2 = sns.kdeplot(X_learnt[:,0], X_learnt[:,1], bw_method=5, color='r', shade=True, label='Learnt',  ax=axes[0], legend=False)
axes[0].set_title('Learning a Gaussian')
handles = [mpatches.Patch(facecolor='b', label="True"),
           mpatches.Patch(facecolor='r', label="Learnt")]
axes[0].legend(handles, ['True', 'Learnt'])

plt.subplot(122)
plt.plot(mu_trace, label='loc trace')
plt.plot(sd_trace, label='sd trace')
plt.legend()

pyro.clear_param_store()
