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
import scipy.stats as st
import configparser


config = configparser.ConfigParser()
config.read('kernel_experiments_config.ini')

def generate_gp_latent(X_star, mean, cov):
    
    return np.random.multivariate_normal(mean(X_star).eval(), cov=cov(X_star, X_star).eval())

def generate_gp_training(X_all, f_all, n_train, noise_var, uniform):
    
    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         pdf = 0.5*st.norm.pdf(X_all, 2, 0.7) + 0.5*st.norm.pdf(X_all, 7.5,1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
     
    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    test_index = [i for i in np.arange(len(X_all)) if i not in train_index]
    X = X_all[train_index]
    f = f_all[train_index]
    X_star = X_all[test_index]
    f_star = f_all[test_index]
    y = f + np.random.normal(0, scale=np.sqrt(noise_var), size=n_train)
    return X, y, X_star, f_star, index

#---------------GP Framework----------------------------------------------
    
def get_kernel(kernel_type, hyper_params):
    
    if kernel_type == 'SE':
        
        sig_var = hyper_params[0]
        lengthscale = hyper_params[1]
        
        return pm.gp.cov.Constant(sig_var)*pm.gp.cov.ExpQuad(1, lengthscale)

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
  

def get_kernel_hyp_string(kernel_type, hyper_params):
      
    if kernel_type == 'SE':
        
        sig_var = hyper_params[0]
        lengthscale = hyper_params[1]
        noise_var = hyper_params[2]
        return r'$\{\sigma_{f}^{2}$: ' + str(sig_var) + r',$\gamma$: ' + str(lengthscale) + r',$\sigma_{n}^{2}$: ' + str(noise_var) + '}'

    elif kernel_type == 'PER':
        
        period = hyper_params[0]
        lengthscale = hyper_params[1]
        return r'$\{p$: ' + str(period) + r',$\gamma$: ' + str(lengthscale) + '}'
        
    elif kernel_type == 'MATERN':
        
        lengthscale = hyper_params[0]
        return r',$\gamma$: ' + str(lengthscale)  + '}'
        
        
    elif kernel_type == 'RQ':
        
        alpha = hyper_params[0]
        lengthscale = hyper_params[0]
        return r'$\{\alpha$: ' + str(alpha) + r',$\gamma$: ' + str(lengthscale) + '}'
    

def get_kernel_matrix_blocks(cov, X, X_star, n_train, noise_var):
    
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
    post_mean = np.dot(K_s.eval().T, alpha)
    post_cov = K_ss.eval() - K_s.eval().T.dot(K_inv.eval()).dot(K_s.eval())
    post_std = np.sqrt(np.diag(post_cov))
    return post_mean, post_cov, post_std
    
def posterior_predictive_samples(post_mean, post_cov):
    
    return np.random.multivariate_normal(post_mean, post_cov, 20)

# Metrics for assessing fit in Regression 

def rmse(post_mean, f_star):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - f_star))),2)

def log_predictive_density(predictive_density):

      return np.round(np.sum(np.log(predictive_density)), 2)

#-------------Plotting----------------------------------------------------

def plot_noisy_data(X, y, X_star, f_star, title):

    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    ax.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    ax.set_xlabel("X"); ax.set_ylabel("The true f(x)") 
    plt.legend()
    plt.title(title, fontsize='x-small')


def plot_kernel_matrix(K, title):
    
    plt.matshow(K)
    plt.colorbar()
    plt.title(title, fontsize='x-small')
    
def plot_prior_samples(X_star, K_ss):
      
      std = np.sqrt(np.diag(K_ss))
      plt.figure()
      plt.plot(X_star,st.multivariate_normal.rvs(cov=K_ss, size=10).T)
      plt.fill_between(np.ravel(X_star), 0 - 1.96*std, 
                     0 + 1.96*std, alpha=0.2, color='r',
                     label='95% CR')

def plot_gp(X_star, f_star, X, y, post_mean, post_std, post_samples, title):
    
    plt.figure()
    if post_samples != []:
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

#    X = np.asarray(pd.read_csv('X.csv', header=None)).reshape(30,1)
#    y = np.asanyarray(pd.read_csv('y.csv', header=None)).reshape(30,)
    
    # Data

    n_train = 20
    n_star = 200
    
    xmin = 0
    xmax = 10
    
    X_all = np.linspace(xmin, xmax+3,n_star)[:,None]
    
    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyperparameters 
    
    sig_var = 5.0
    lengthscale = 1.0
    noise_var = 0.2
    hyp = [sig_var, lengthscale, noise_var]
    cov = get_kernel('SE', [sig_var, lengthscale])
    hyp_string = get_kernel_hyp_string('SE', [sig_var, lengthscale, noise_var])
    
    f_all = generate_gp_latent(X_all, mean, cov)
    
    uniform = True
    X, y, X_star, f_star, index = generate_gp_training(X_all, f_all, n_train, noise_var, uniform)
    K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, noise_var)
    
    # Add very slight perturbation to the covariance matrix diagonal to improve numerical stability
    
    K_stable = K + 1e-12 * tt.eye(n_train)
    
    # Generate prior samples and plot covariance matrix to verify if the Data scheme is correct
    
    plot_kernel_matrix(K_ss.eval(), 'Cov. matrix with true values ' + get_kernel_hyp_string('SE',hyp))
    plot_prior_samples(X_star, K_ss.eval())
    
    #---------------------------------------------------------------------
    # Analytically compute posterior mean and posterior covariance
    # Algorithm 2.1 in Rasmussen and Williams
    #---------------------------------------------------------------------
    
    # Plot the data, kernel matrix and the unobserved latent function
    
    plot_noisy_data(X, y, X_star, f_star, 'Data + iid noise')
    
    # Note: The below function returns theano variables that need to be evaluated to draw values
    post_mean, post_cov, post_std = analytical_gp(y, K_stable, K_s, K_ss, K_noise, K_inv)
    post_samples = posterior_predictive_samples(post_mean, post_cov)
    rmse_ = rmse(post_mean, f_star)
    lpd_ = log_predictive_density(st.multivariate_normal.pdf(f_star, post_mean, post_cov, allow_singular=True))
    
    kernel = 'Kernel: 5.0* RBF(lenghtscale = 1) + WhiteKernel(noise_level=0.2)'
    title = 'GPR' + '\n' + kernel + '\n' + 'RMSE: ' + str(rmse_) '\n' + 'LPD: ' + str(lpd_)
    plot_gp(X_star, f_star, X, y, post_mean, post_std, [], title)
    plot_kernel_matrix(post_cov.eval(),'')
    
    #---------------------------------------------------------------------
    
    # Type II ML for hyp.
    
    #---------------------------------------------------------------------
    
    kernel = Ck(10.0, (1e-10, 1e2)) * RBF(2, length_scale_bounds=(0.5, 8)) + WhiteKernel(10.0, noise_level_bounds=(1e-5,100))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        
    # Fit to data 
    gpr.fit(X, y)        
    post_mean, post_cov = gpr.predict(X_star, return_cov = True) 
    post_std = np.sqrt(np.diag(post_cov))
    post_samples = np.random.multivariate_normal(post_mean, post_cov, 10)
    rmse_ = rmse(post_mean, f_star)
    lpd_ = log_predictive_density(st.multivariate_normal.pdf(f_star, post_mean, post_cov, allow_singular=True))
    title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_) 
    plot_gp(X_star, f_star, X, y, post_mean, post_std, [], title)
    #plot_kernel_matrix(post_cov,'')
    
    plot_lml_surface_3way(gpr, sig_var)

    ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
    ml_deltas_dict = {'lengthscale': ml_deltas[1], 'noise_var': ml_deltas[2], 'sig_var': ml_deltas[0]}

    #---------------------------------------------------------------------
    
    # Vanilla GP - pymc3
    
    #---------------------------------------------------------------------

    with pm.Model() as model:
        
        sig_var = ml_deltas[0]
        lengthscale = ml_deltas[1]
        noise_var = ml_deltas[2]
    
        # Specify the covariance function.
        cov_func = pm.gp.cov.Constant(sig_var)*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.Marginal(cov_func=cov_func)
            
        # Marginal Likelihood
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
        
    with model:
        f_cond = gp.conditional("f_cond", Xnew=X_star)
        #pred_samples = pm.sample_posterior_predictive(vars=[f_cond], samples=10)


post_pred_mean, post_pred_cov = gp.predict(X_star, pred_noise=True)
post_pred_mean, post_pred_cov_nf = gp.predict(X_star, pred_noise=False)
post_pred_std = np.sqrt(np.diag(post_pred_cov))

rmse_ = rmse(post_pred_mean, f_star)
lpd_ = log_predictive_density(st.multivariate_normal.pdf(f_star, post_pred_mean, post_pred_cov, allow_singular=True))

pred_samples = posterior_predictive_samples(post_pred_mean, post_pred_cov_nf)
title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_) 
plot_gp(X_star, f_star, X, y, post_pred_mean, post_pred_std, pred_samples,title)
    
#-----------------------------------------------------

#       Hybrid Monte Carlo
    
#-----------------------------------------------------

  with pm.Model() as hmc_gp_model:
        
       # prior on lengthscale 
       lengthscale = pm.Gamma('lengthscale', alpha=2, beta=1)
       
       sig_var = pm.HalfCauchy('sig_var', beta=3)
       
       #prior on noise variance
       noise_var = pm.HalfCauchy('noise_var', beta=5)
       
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
                
       trace = pm.sample(draws=500)
              
  with hmc_gp_model:
       y_pred = gp.conditional("y_pred", X_star)
       
# Box standard Traceplot on log axis with deltas and means highlighted
       
ml_deltas_dict = {lengthscale: ml_deltas[1], noise_var: ml_deltas[2], sig_var: ml_deltas[0]}
varnames = ['sig_var', 'lengthscale','noise_var']
hyp_map = np.round(get_trace_means(trace, varnames),3)
priors = [lengthscale.distribution, sig_var.distribution, noise_var.distribution]
traces = pm.traceplot(trace, varnames=[lengthscale, noise_var, sig_var], priors=priors, prior_style='--', lines=ml_deltas_dict, bw=2, combined=True)
traces[0][0].axvline(x=hyp_map[1], color='b',alpha=0.5, label='HMC ' + str(hyp_map[1]))
traces[1][0].axvline(x=hyp_map[2], color='b', alpha=0.5, label='HMC ' + str(hyp_map[2]))
traces[2][0].axvline(x=hyp_map[0], color='b', alpha=0.5, label='HMC ' + str(hyp_map[0]))
traces[0][0].axvline(x=ml_deltas[1], color='r',alpha=0.5, label='ML ' + str(ml_deltas[1]))
traces[1][0].axvline(x=ml_deltas[2], color='r', alpha=0.5, label='ML ' + str(ml_deltas[2]))
traces[2][0].axvline(x=ml_deltas[0], color='r', alpha=0.5, label='ML ' + str(ml_deltas[0]))
traces[0][0].axvline(x=1.0, color='g',alpha=0.5, label='True ' + str(1.0))
traces[1][0].axvline(x=0.2, color='g', alpha=0.5, label= 'True ' + str(0.2))
traces[2][0].axvline(x=5.0, color='g', alpha=0.5, label='True ' + str(5.0))
traces[0][1].axhline(y=hyp_map[1], color='b', alpha=0.5)
traces[1][1].axhline(y=hyp_map[2], color='b', alpha=0.5)
traces[2][1].axhline(y=hyp_map[0], color='b', alpha=0.5)
traces[0][0].axes.set_xscale('log')
traces[1][0].axes.set_xscale('log')
traces[2][0].axes.set_xscale('log')
traces[0][0].hist(trace['lengthscale'], bins=100, normed=True, color='orange', alpha=0.3)
traces[1][0].hist(trace['noise_var'], bins=100, normed=True, color='orange', alpha=0.3)
traces[2][0].hist(trace['sig_var'], bins=100, normed=True, color='orange', alpha=0.3)
for j, k in [(0,0), (1,0), (2,0)]:
      traces[j][k].legend(fontsize='x-small')

# Write out trace summary & autocorrelation plots

prefix = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/'
summary_df = pm.summary(trace)
summary_df['Acc Rate'] = np.mean(trace.get_sampler_stats('mean_tree_accept'))
np.round(summary_df,3).to_csv(prefix + 'trace_summary_se_unif.csv')
pm.autocorrplot(trace)

# Compute posterior predictive mean and covariance - careful (not so obvious)

def get_combined_trace(trace):
      
    trace_df = pd.DataFrame()
    trace_df['sig_var'] = np.mean(trace.get_values('sig_var', combine=False), axis=0)
    trace_df['lengthscale'] = np.mean(trace.get_values('lengthscale', combine=False), axis=0)
    trace_df['noise_var'] = np.mean(trace.get_values('noise_var', combine=False), axis=0)
    return trace_df

def get_trace_means(trace, varnames):
      
      trace_means = []
      for i in varnames:
            trace_means.append(trace[i].mean())
      return trace_means

def get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv):
      
      #L = np.linalg.cholesky(K_noise.eval())
      #alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
      #post_mean_trace = np.dot(K_s.T.eval(), alpha)
      return  K_s.T.dot(K_inv).dot(y)

def get_post_cov_theta(theta, K_s, K_ss, K_inv):
      
     return K_ss - K_s.T.dot(K_inv).dot(K_s)

def get_joint_value_from_trace(trace, varnames, i):
      
      joint = []
      for v in varnames:
            joint.append(trace[v][i])
      return joint

def get_post_mcmc_mean_cov(trace_df, X, X_star, varnames, n_train, test_size):
      
      #sum_means = np.zeros(test_size)
      sum_means = []
      #sum_cov = np.zeros((test_size, test_size))
      sum_cov = []
      #sum_mean_sq = np.zeros((test_size, test_size))
      for i in range(0,len(trace_df)):
            print(i)
            theta = get_joint_value_from_trace(trace_df, varnames, i) 
            cov = get_kernel('SE', [theta[0], theta[1]])
            K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, theta[2])
            print('Done at ' + str(i))
            mean_i = get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv)
            cov_i = get_post_cov_theta(theta, K_s, K_ss, K_inv)
            print('Done here too')
            sum_means.append(mean_i)
            sum_cov.append(cov_i)
            #sum_mean_sq += np.outer(mean_i.eval(), mean_i.eval())
            #sum_means += mean_i.eval()
            #sum_cov += cov_i.eval()
            print('Done adding too')
      post_mean_trace = sum_means/len(trace_df)
      post_cov_mean =  sum_cov/len(trace_df) + sum_mean_sq/len(trace_df)  - np.outer(post_mean_trace, post_mean_trace)
      return post_mean_trace, post_cov_mean
     
varnames = ['sig_var', 'lengthscale','noise_var']
theta = get_trace_means(trace_df, varnames=['sig_var', 'lengthscale','noise_var'])

test_size=180
trace_df = get_combined_trace(trace)
post_mean_trace, post_cov_mean = get_post_mcmc_mean_cov(trace_df, X, X_star, varnames, n_train, test_size)
post_s_std = np.sqrt(np.diag(post_cov_mean))

# Plot fit with mean and variance ML case

plt.figure()
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
plt.plot(X_star, f_star, "dodgerblue", lw=1.0, label="True f",alpha=0.7);
plt.plot(X_star, post_pred_mean, color='r', label=r'$\mu_{*}^{II ML}$')
plt.legend(fontsize='x-small')
plt.fill_between(np.ravel(X_star), post_pred_mean - 2*post_pred_std, 
                     post_pred_mean + 2*post_pred_std, alpha=0.2, color='r',
                     label=r'$2\sigma_{*}^{2(II ML)}$')
post_samples = posterior_predictive_samples(post_pred_mean, post_pred_cov_nf)
plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
plt.title('Type II ML' + '\n' 
          'True values: ' + get_kernel_hyp_string('SE', [5.0, 1.0, 1.0])  + '\n' +
          'ML: ' + get_kernel_hyp_string('SE', np.round(ml_deltas, 3))
          , fontsize='x-small')
plt.legend(fontsize='x-small')

# Plot fit with mean and variance HMC case

plt.figure()
plt.plot(X_star, post_mean_trace, color='b', label=r'$\mu_{*}^{HMC}$')
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
plt.plot(X_star, f_star, "dodgerblue", lw=1.0, label="True f",alpha=0.7);
plt.fill_between(np.ravel(X_star), (post_mean_trace - 2*post_s_std).reshape(test_size,), 
                     (post_mean_trace + 2*post_s_std).reshape(test_size,), alpha=0.2, color='b',
                     label=r'$2\sigma_{*}^{2 (HMC)}$')
plt.legend(fontsize='x-small')
plt.title('HMC' + '\n' 
          'True values: ' + get_kernel_hyp_string('SE', [5.0, 1.0, 1.0])  + '\n' +
          'HMC: ' + get_kernel_hyp_string('SE', np.round(theta, 3))
          , fontsize='x-small')
post_samples = posterior_predictive_samples(post_mean_trace, post_cov_mean)
plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)

# Plot fit with mean and variance MAP case

K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, theta[2])
post_mean_map = get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv)
post_cov_map = get_post_cov_theta(theta, K_s, K_ss, K_inv)
post_std_map = np.sqrt(np.diag(post_cov_map.eval()))

plt.plot(X_star, post_mean_map, color='g', label=r'$\mu_{*}^{MAP}$')
plt.fill_between(np.ravel(X_star), (post_mean_map - 2*post_std_map).reshape(test_size,), 
                     (post_mean_map + 2*post_std_map).reshape(test_size,), alpha=0.2, color='g',
                     label=r'$2\sigma_{*}^{2 (MAP)}$')




# Fit metrics

rmse_mcmc = rmse(f_star, post_mean_trace)
rmse_map = rmse(f_star, post_mean_trace)
rmse_ml = rmse(f_star, post_pred_mean)
lpd_ = log_predictive_density(st.multivariate_normal.pdf(f_star, post_mean_trace, ))





