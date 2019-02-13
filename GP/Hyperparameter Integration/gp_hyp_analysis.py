# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
from theano.tensor.nlinalg import matrix_inverse, det
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from matplotlib.colors import LogNorm


def generate_gp_latent(X_star, K_ss):
    
    return np.random.multivariate_normal(mean(X_star).eval(), cov=K_ss.eval())

def generate_gp_training(X_star, f_star, n, xmax, noise_var):
    
    X_index = np.random.randint(0,xmax,30)
    X = X_star[X_index]
    f = f_star[X_index]
    y = f + np.random.normal(0, scale=np.sqrt(noise_var), size=n)
    return X, y

#---------------GP Framework--------------------------------------------------
    
    
def get_kernel(kernel_type, hyper_params):
    
    if kernel_type == 'SE':
        
        sig_var = hyper_params[0]
        lengthscale = hyper_params[1]
        
        return sig_var*pm.gp.cov.ExpQuad(1, lengthscale)

    elif kernel_type == 'PER':
        
        period = hyper_params[0]
        lengthscale = hyper_params[1]
        
        return pm.gp.cov.Periodic(1,period, ls=lengthscale)
        
    
    elif kernel_type == 'MATERN':
        
        lengthscale = hyper_params[0]
        
        return pm.gp.Matern32(1,ls=lengthscale)
        
        
    elif kernel_type == 'RQ':
        
        alpha = hyper_params[0]
        lengthscale = hyper_params[0]
        
        return pm.gp.cov.RatQuad(1, alpha, ls=lengthscale)
    

def get_kernel_matrix_blocks(X, X_star, n_train):
    
    K = cov(X)
    K_s = cov(X, X_star)
    K_ss = cov(X_star, X_star)
    K_noise = K + noise_var*tt.eye(n_train)
    K_inv = matrix_inverse(K_noise)
    
    return K, K_s, K_ss, K_noise, K_inv

def analytical_gp(y, K, K_s, K_ss, K_noise, K_inv):
    
    L = np.linalg.cholesky(K_noise.eval())
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    #v = np.linalg.solve(L, K_s.eval())
    post_mean = np.dot(K_s.T.eval(), alpha)
    post_cov = K_ss.eval() - K_s.T.dot(K_inv).dot(K_s)
    post_std = np.sqrt(np.diag(post_cov.eval()))
    
    return post_mean, post_cov, post_std
    

def posterior_predictive_samples(post_mean, post_cov):
    
    return np.random.multivariate_normal(post_mean, post_cov.eval(), 10)


def compute_rmse(post_mean, f_star):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - f_star))),2)


#-------------Plotting----------------------------------------------------

def plot_noisy_data(y, X_star, f_star, title):

    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    ax.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    ax.set_xlabel("X"); ax.set_ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')


def plot_kernel_matrix(K, title):
    
    plt.figure()
    plt.matshow(K)
    plt.colorbar()
    plt.title(title, fontsize='x-small')


def plot_gp(X_star, f_star, X, y, samples, post_mean, post_cov, post_samples, title):
    
    plt.figure()
    plt.plot(X_star, post_samples.T, color='orange', alpha=0.5)
    plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'bx', ms=3, alpha=0.8, color='r');
    plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), post_mean - 1.96*post_std, 
                     post_mean + 1.96*post_std, alpha=0.2, color='r',
                     label='95% CR')
    plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')
    
    
def plot_lml_surface_3way(gpr, sig_var):
    
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    l_log = np.logspace(-2, 3, 100)
    noise_log  = np.logspace(-2, 2, 100)
    l_log_mesh, noise_log_mesh = np.meshgrid(l_log, noise_log)
    LML = [[gpr.log_marginal_likelihood(np.log([sig_var, l_log_mesh[i, j], noise_log_mesh[i, j]]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, noise_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, noise_var, 'rx')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise-level")
    
    plt.subplot(132)
    l_log = np.logspace(-2, 3, 100)
    signal_log  = np.logspace(-3, 5, 100)
    l_log_mesh, signal_log_mesh = np.meshgrid(l_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([signal_log_mesh[i,j], l_log_mesh[i, j], noise_var]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(l_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(lengthscale, sig_var, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Signal-Var")
    
    plt.subplot(133)
    noise_log = np.logspace(-2, 3, 100)
    signal_log  = np.logspace(-3, 5, 100)
    noise_log_mesh, signal_log_mesh = np.meshgrid(noise_log, signal_log)
    LML = [[gpr.log_marginal_likelihood(np.log([signal_log_mesh[i,j], lengthscale, noise_log_mesh[i,j]]))
            for i in range(l_log_mesh.shape[0])] for j in range(l_log_mesh.shape[1])]
    LML = np.array(LML).T
    
    vmin, vmax = (-LML).min(), (-LML).max()
    #vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=1)
    plt.contourf(noise_log_mesh, signal_log_mesh, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cm.get_cmap('jet'))
    plt.plot(noise_var, sig_var, 'rx')
    #plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise-level")
    plt.ylabel("Signal-Var")
    
    plt.suptitle('LML Surface ' + '\n' + str(gpr.kernel_), fontsize='small')



if __name__ == "__main__":


    # If pre-loading 

    X_star = np.linspace(0,13,1000)[:,None]
    f_star = np.asarray(pd.read_csv('f_star.csv', header=None)).reshape(1000,)
    X = np.asarray(pd.read_csv('X.csv', header=None)).reshape(30,1)
    f = np.asarray(pd.read_csv('f.csv', header=None)).reshape(30,)
    y = np.asanyarray(pd.read_csv('y.csv', header=None)).reshape(30,)

    # Data

    n_train = 30
    n_star = 1000
    
    xmin = 0
    xmax = 13
    
    X_star = np.linspace(xmin, xmax,n_star)[:,None]
    
    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyper-parameters 
    
    sig_var = 2.0
    lengthscale = 1.0
    noise_var = 0.2
    
    cov = get_kernel('SE', [sig_var, lengthscale])
    
    K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train)
    
    # Add very slight perturbation to the covariance matrix diagonal to improve numerical stability
    K_stable = K + 1e-12 * tt.eye(n_train)
    
    
    f_star = generate_gp_latent(X_star, K_ss)
    X, y = generate_gp_training(X_star, f_star, n_train, 800, noise_var)
    
    #---------------------------------------------------------------------
    # Analytically compute posterior mean and posterior covariance
    # Algorithm 2.1 in Rasmussen and Williams
    #---------------------------------------------------------------------
    
    # Plot the data, kernel matrix and the unobserved latent function
    
    plot_noisy_data(y, X_star, f_star, 'Data + iid noise')
    plot_kernel_matrix(K_ss.eval(), 'Kernel with true values ' + r'$\{\sigma_{f}^{2}\}$')
    
    post_mean, post_cov, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
    post_samples = posterior_predictive_samples(post_mean, post_cov)
    rmse_ = compute_rmse(post_mean, f_star)
    
    kernel = 'Kernel: 2.0* RBF(lenghtscale = 1) + WhiteKernel(noise_level=0.2)'
    title = 'GPR' + '\n' + kernel + '\n' + 'RMSE: ' + str(rmse_)
    plot_gp(X_star, f_star, X, y, post_samples, post_mean, post_cov, post_samples, title)

    #---------------------------------------------------------------------
    
    # Type II ML for hyp.
    
    #---------------------------------------------------------------------
    
    kernel = Ck(100.0, (1e-10, 1e3)) * RBF(2, length_scale_bounds=(0.5, 3)) + WhiteKernel(0.01, noise_level_bounds=(1e-5,10))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
    # Fit to data 
    gpr.fit(X, y)        
    
    post_mean, post_std = gpr.predict(X_star, return_std = True)    
    post_samples = np.random.multivariate_normal(post_mean, post_cov.eval(), 10)
    rmse_ = compute_rmse(post_mean, f_star)
    
    title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_)
    plot_gp(X_star, f_star, X, y, post_samples, post_mean, post_cov, post_samples, title)

    #---------------------------------------------------------------------
    
    # Vanilla GP - pymc3
    
    #---------------------------------------------------------------------

    with pm.Model() as model:
        
        sig_var = 2.25
        lengthscale = 1.48
        noise_var = 0.19
    
        # Specify the covariance function.
        cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.Latent(cov_func=cov_func)
    
        # Place a GP prior over the function f.
        f = gp.prior("f", X=X)
        
        # Likelihood
        y_obs = pm.MvNormal('y', mu=f, cov=noise_var * tt.eye(n_train), shape=n_train, observed=y)        
        
        trace = pm.sample()
  
    with model:
        
        f_cond = gp.conditional("f_cond", Xnew=X_star)
        post_pred_mean = f_cond.mean()
        pred_samples = pm.sample_ppc(trace, vars=[f_cond], samples=100)
        
post_samples = pred_samples['f_cond']
title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_)

def plot_base_ml2_pymc3(X_star, f_star, X, y, post_mean, post_samples, title):
        
        fig = plt.figure()
        ax = fig.gca()
        plot_gp_dist(ax, pred_samples['f_cond'], X_star);
        plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
        plt.plot(X, y, 'bx', ms=3, alpha=0.8)
        plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
        plt.title(title, fontsize='x-small')
        plt.legend(fontsize='x-small')
        #plt.plot(X_star, np.mean(pred_samples['f_cond'].T, axis=1), color='r', lw=2, label='Posterior mean 2')

        plot_base_ml2_pymc3(X_star, f_star, X, y, post_mean, post_samples, title)

    #TODO: Figure out how to get post mean / post cov from f_cond in pymc3 framework
    
    #-----------------------------------------------------

    #       Hybrid Monte Carlo
    
    #-----------------------------------------------------


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


# HMC Sampling -> all go


   with pm.Model() as hmc_gp_model:
        
       # prior on lengthscale 
       l = pm.Gamma('lengthscale', alpha=2, beta=1)
       
       sig_var = pm.HalfCauchy('sig_var', beta=3)
       
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
       
       
ml_deltas = {l: 1.48,
             noise_var: 0.19,
             sig_var: 2.25
             }
priors = [l.distribution, sig_var.distribution, noise_var.distribution]

pm.traceplot(trace, varnames=[l, noise_var, sig_var], priors=priors, lines=ml_deltas, bw=4, combined=True)

post_pred_mean = np.mean(pred_samples["y_pred"].T, axis=1)

rmse_ = compute_rmse(f_star, post_pred_mean)


title = "GPR $f(x)$ w. HMC sampling for " + r'$\{\sigma_{f}^{2}\, \sigma_{n}^{2}, l\}$' + '\n' + \
                                    'RMSE: ' + str(rmse_)
                                    

fig = plt.figure() 
ax = fig.gca()
plot_gp_dist(ax, pred_samples["y_pred"], X_star);
plt.plot(X_star, post_pred_mean, color='g', lw=2, label='Posterior mean')
plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7)
plt.plot(X, y, 'bx', ms=3, alpha=0.8)
plt.title(title, fontsize='x-small')
plt.legend(fontsize='x-small')


