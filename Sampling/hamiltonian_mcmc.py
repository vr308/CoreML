#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:44:56 2019

@author: vidhi
"""

# Sampling with pymc3 
import pymc3 as pm
import seaborn as sns
from sampled import sampled  
import scipy.stats as st  
      
def tt_donut_pdf(scale):
    """Compare to `donut_pdf`"""
    def logp(x):
         return -tt.square((1 - x.norm(2)) / scale)
    return logp

@sampled
def donut(scale=0.1, **observed):
    """Gets samples from the donut pdf, and allows adjusting the scale of the donut at sample time."""
    pm.DensityDist('donut', logp=tt_donut_pdf(scale), shape=2, testval=[0, 1])
      
    
with donut(scale=0.005):
    metropolis_sample = pm.sample(draws=1000, step=pm.Metropolis())
    hmc_sample1 = pm.sample(draws=1000, step=pm.HamiltonianMC(path_length=10))
    hmc_sample2 = pm.sample(draws=1000, step=pm.HamiltonianMC(path_length=2))

def jointplot(ary):
    """Helper to plot everything consistently"""
    sns.jointplot(*ary.T, alpha=0.1, stat_func=None, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    
jointplot(metropolis_sample.get_values('donut'))
jointplot(hmc_sample1.get_values('donut'))
jointplot(hmc_sample2.get_values('donut'))

#  Verifying the donut pdf

scale=0.01
donut_density = lambda x: np.exp(-((1 - np.linalg.norm(x)) / scale)**2)
x1 = np.linspace(-1,1,100)
x2 = np.linspace(-1,1,100)

X1, X2 = np.meshgrid(x1,x2)

points = np.vstack([X1.ravel(), X2.ravel()]).T

pdf = []
for i in points:
      pdf.append(donut_density(i))

plt.figure()
plt.contourf(X1, X2, np.array(pdf).reshape(100,100))

# Simple example of HMC with leapfrog paths overlaid on the joint density 

def gen_data(position, momentum, n=10000):
    """Generate a background density plot for the position and momentum distributions.  Not used for sampling."""
    q = position.rvs(n)
    p = momentum(q).rvs()
    return np.column_stack([q, p])

def leapfrog(q, p, dHdq, dHdp, step_size, n_steps):
    """Perform the leapfrog integration.  
    
    Similar to the implementations in PyMC3, but 
    returns an array of all the points in the path
    for plotting.  It is a pretty general 
    implementation, in that it does hardcode
    the potential or kinetic energies.
    
    Args:
        q: current position
        p: current momentum
        dHdq: see Hamilton's equations above
        dHdp: see Hamilton's equations above
        step_size: How big of a step to take
        n_steps: How many steps to take
    
    Returns:
        (`n_steps` x 2) numpy array showing the 
        path through (position, momentum) space 
        the Hamiltonian path took.
    """
    data = [[q, p]]
    p += 0.5 * step_size * -dHdq(q, p)
    q += step_size * dHdp(q, p)
    data.append([q, p])
    for _ in range(n_steps - 1):
        p += step_size * -dHdq(q, p)
        q += step_size * dHdp(q, p)
        data.append([q, p])
    p += 0.5 * step_size * -dHdq(q, p)
    return np.array(data)    

def leapfrog_paths(position, momentum, dHdq, dHdp, n=10):
    """Get a number `n` of paths from the HMC sampler.

    This is not quite as general -- I hardcode a step 
    size of 0.01 and 100 steps, since a path length of 1
    looked pretty good (and *not* because it is the best
    choice).  Returns an iterator of plot data.
    """
    q = position.rvs()
    p = momentum(q).rvs()
    for _ in range(n):
        path = leapfrog(q, p, dHdq, dHdp, 0.01, 100)
        yield path
        q, _ = path[-1]
        p = momentum(q).rvs()
        

position = st.norm(0, 1)
momentum = lambda q: st.norm(0, np.ones(q.shape))
dHdp = lambda q, p: 2 * p
dHdq = lambda q, p: 2 * q

#  First plot a KDE of the joint pdf
X = gen_data(position, momentum)
g = sns.jointplot(*X.T, kind='kde', stat_func=None, xlim=(-3, 3), ylim=(-3, 3), alpha=0.5)

#  Now plot the Leapfrog paths on top of that
for path in leapfrog_paths(position, momentum, dHdq, dHdp, 15):
    g.ax_joint.plot(*path.T, linewidth=3)