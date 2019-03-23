#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:55:25 2019

@author: vidhi
"""

import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import arviz as arv

sns.set()  # Nice plot aesthetic
np.random.seed(101)

def bayesian_diagnostics_report(trace, fit, inference_blocks, varnames):
      
      print(fit.stansummary(pars=varnames, probs=[0.025,0.975]))

      arv.plot_posterior(fit,var_names=varnames, credible_interval=0.95, bw=2, kind='hist')
      arv.plot_trace(fit, var_names=varnames)
      arv.plot_autocorr(fit, var_names=varnames, combined=True)
      arv.plot_parallel(fit, var_names=varnames)
      arv.plot_ppc(inference_blocks, data_pairs={'y':'y_star'})
      
      g = sns.PairGrid(trace[varnames], palette="Set2")
      g = g.map_diag(sns.kdeplot, lw=1, legend=False)
      g = g.map_upper(plt.scatter, s=0.5)
      g = g.map_lower(sns.kdeplot, cmap="Blues", n_levels=20, shade=True)
      

#-----------------------------------------------------------------------------
# 1d linear Regression 
#-----------------------------------------------------------------------------
      
linear_model = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    vector[N] x_star;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {

    y ~ normal(alpha + beta * x, sigma);
}

generated quantities{

      vector[N] y_star;
      for (n in 1:N){
                  y_star[n] = normal_rng(alpha + beta*x[n], sigma);
      }

}
"""

# Parameters to be inferred

alpha = 4.0
beta = 0.5
sigma = 1.0

# Generate and plot data

x = 10 * np.random.rand(100)
x_star = 10 * np.random.rand(100)
y = alpha + beta * x
y = np.random.normal(y, scale=sigma)

# Put our data in a dictionary
data = {'N': len(x), 'x': x, 'y': y, 'x_star' : x_star}

# Compile the model
sm = pystan.StanModel(model_code=linear_model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)

trace = fit.to_dataframe()[fit.flatnames]

inference_blocks = arv.from_pystan(posterior=fit, posterior_predictive='y_star', observed_data='y')

# Plots using arv - cool package!!

varnames = ['alpha', 'beta', 'sigma']

bayesian_diagnostics_report(trace, fit, inference_blocks, varnames=varnames)

#-----------------------------------------------------------------------------
# Beta Binomial model
#-----------------------------------------------------------------------------

coin_model = """
data{
     int<lower=0> n;
     int<lower=0> y;
}

parameters{
     real<lower=0, upper=1> p;
}

model{
    p ~ beta(2,2);
    y ~ binomial(n, p);
}
"""

y = st.binom.rvs(n=1, p=0.5, size=100)

data = {'n':100, 'y': 60}

sm = pystan.StanModel(model_code=coin_model)
fit = sm.sampling(data=data, iter=1000, chains=2, warmup=500, thin=1, seed=101)
trace = fit.to_dataframe()[fit.flatnames]


#-----------------------------------------------------------------------------
# Estimating mu and sd of a normal
#-----------------------------------------------------------------------------
 

