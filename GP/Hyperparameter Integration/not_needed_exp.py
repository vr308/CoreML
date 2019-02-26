#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:55:30 2019

@author: vidhi
"""


    # HMC Sampling -> lengthscale
    
    with pm.Model() as hmc_gp_model:
        
       # prior on lengthscale 
       l = pm.Gamma('lengthscale', alpha=2, beta=1)
             
       #sig_var = pm.HalfCauchy('sig_var', beta=5)
       sig_var = 2.25
       
       #prior on noise variance
       noise_var = 0.19
       
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=l)
    
       # Specify the GP.  The default mean function is `Zero`.
       gp = pm.gp.Latent(cov_func=cov_func)
    
       # Place a GP prior over the function f.
       f = gp.prior("f", X=X)
        
       # Likelihood
       y_obs = pm.MvNormal('y', mu=f, cov=noise_var * tt.eye(n_train), shape=n_train, observed=y)        
         
       
       step = pm.hmc.HamiltonianMC(path_length=2.0, 
                                   adapt_step_size=True, 
                                   gamma = 0.05, 
                                   k = 0.75)
       
       trace = pm.sample(step=step, tune=200)
              
    with hmc_gp_model:
       y_pred = gp.conditional("y_pred", X_star)
       
    with hmc_gp_model:
       pred_samples = pm.sample_ppc(trace, vars=[y_pred,l], samples=30)


ml_deltas = {l: gpr.kernel_.k1.k2.length_scale,
             noise_var: gpr.kernel_.k2.noise_level,
             sig_var: gpr.kernel_.k1.k1
             }
pm.traceplot(trace, varnames=[l], priors=[l.distribution], lines=ml_deltas, bw=4, combined=True)

post_pred_mean = np.mean(pred_samples["y_pred"].T, axis=1)

rmse_ = compute_rmse(f_star, post_pred_mean)

# Plotting

title = "GPR $f(x)$ w. HMC sampling for " + r'$\{l\}$' + '\n' + \
                                    r'$\{\sigma_{f}^{2}, \sigma_{n}^{2}\} = {2.25, 0.19}$' + '\n' + \
                                    'RMSE: ' + str(rmse_)
                                    
   
fig = plt.figure() 
ax = fig.gca()
plot_gp_dist(ax, pred_samples["y_pred"], X_star);
plt.plot(X_star, np.mean(pred_samples["y_pred"].T, axis=1), color='g', lw=2, label='Posterior mean')
plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7)
plt.plot(X, y, 'bx', ms=3, alpha=0.8)
plt.title(title, fontsize='x-small')
plt.legend(fontsize='x-small')


# HMC Sampling -> noise var


    with pm.Model() as hmc_gp_model:
        
       # prior on lengthscale 
       #l = pm.Gamma('lengthscale', alpha=2, beta=1)
       l = 1.48
       
       #sig_var = pm.HalfCauchy('sig_var', beta=5)
       sig_var = 2.25
       
       #prior on noise variance
       noise_var = pm.HalfCauchy('noise_var', beta=5)
       
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=l)
    
       # Specify the GP.  The default mean function is `Zero`.
       gp = pm.gp.Latent(cov_func=cov_func)
    
       # Place a GP prior over the function f.
       f = gp.prior("f", X=X)
        
       # Likelihood
       y_obs = pm.MvNormal('y', mu=f, cov=noise_var * tt.eye(n_train), shape=n_train, observed=y)        
         
       
       step = pm.hmc.HamiltonianMC(path_length=2.0, 
                                   adapt_step_size=True, 
                                   gamma = 0.05, 
                                   k = 0.75)
       
       trace = pm.sample(step=step, tune=200)
              
    with hmc_gp_model:
       y_pred = gp.conditional("y_pred", X_star)
       
    with hmc_gp_model:
       pred_samples = pm.sample_ppc(trace, vars=[y_pred], samples=30)


ml_deltas = {l: gpr.kernel_.k1.k2.length_scale,
             noise_var: gpr.kernel_.k2.noise_level,
             sig_var: gpr.kernel_.k1.k1
             }
pm.traceplot(trace, varnames=[noise_var], priors=[noise_var.distribution], lines={noise_var: gpr.kernel_.k2.noise_level}, bw=4, combined=True)

post_pred_mean = np.mean(pred_samples["y_pred"].T, axis=1)

rmse_ = compute_rmse(f_star, post_pred_mean)


title = "GPR $f(x)$ w. HMC sampling for " + r'$\{\sigma_{n}^{2}\}$' + '\n' + \
                                    r'$\{\sigma_{f}^{2}, l\} = {2.25, 1.48}$' + '\n' + \
                                    'RMSE: ' + str(rmse_)
                                    

fig = plt.figure() 
ax = fig.gca()
plot_gp_dist(ax, pred_samples["y_pred"], X_star);
plt.plot(X_star, post_pred_mean, color='g', lw=2, label='Posterior mean')
plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7)
plt.plot(X, y, 'bx', ms=3, alpha=0.8)
plt.title(title, fontsize='x-small')
plt.legend(fontsize='x-small')


# HMC Sampling -> signal var


  with pm.Model() as hmc_gp_model:
        
       # prior on lengthscale 
       #l = pm.Gamma('lengthscale', alpha=2, beta=1)
       l = 1.48
       
       #sig_var = pm.HalfCauchy('sig_var', beta=5)
       sig_var = pm.HalfCauchy('sig_var', beta=3)
       
       #prior on noise variance
       #noise_var = pm.HalfCauchy('noise_var', beta=5)
       noise_var = 0.19
       
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=l)
    
       # Specify the GP.  The default mean function is `Zero`.
       gp = pm.gp.Latent(cov_func=cov_func)
    
       # Place a GP prior over the function f.
       f = gp.prior("f", X=X)
        
       # Likelihood
       y_obs = pm.MvNormal('y', mu=f, cov=noise_var * tt.eye(n_train), shape=n_train, observed=y)        
         
       
       step = pm.hmc.HamiltonianMC(path_length=2.0, 
                                   adapt_step_size=True, 
                                   gamma = 0.05, 
                                   k = 0.75)
       
       trace = pm.sample(step=step, tune=200)
              
    with hmc_gp_model:
       y_pred = gp.conditional("y_pred", X_star)
       
    with hmc_gp_model:
       pred_samples = pm.sample_ppc(trace, vars=[y_pred], samples=30)


ml_deltas = {l: 1.48,
             noise_var: 0.19,
             sig_var: 2.25
             }

pm.traceplot(trace, varnames=[sig_var], priors=[sig_var.distribution], lines={sig_var: 2.25}, bw=4, combined=True)

post_pred_mean = np.mean(pred_samples["y_pred"].T, axis=1)

rmse_ = compute_rmse(f_star, post_pred_mean)



title = "GPR $f(x)$ w. HMC sampling for " + r'$\{\sigma_{f}^{2}\}$' + '\n' + \
                                    r'$\{\sigma_{n}^{2}, l\} = {0.19, 1.48}$' + '\n' + \
                                    'RMSE: ' + str(rmse_)
                                    

fig = plt.figure() 
ax = fig.gca()
plot_gp_dist(ax, pred_samples["y_pred"], X_star);
plt.plot(X_star, post_pred_mean, color='g', lw=2, label='Posterior mean')
plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7)
plt.plot(X, y, 'bx', ms=3, alpha=0.8)
plt.title(title, fontsize='x-small')
plt.legend(fontsize='x-small')