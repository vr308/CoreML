# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import scipy.stats as st

import warnings
warnings.filterwarnings("ignore")



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
    return X, y, X_star, f_star, f, train_index

#---------------------GP Framework---------------------------------------------
    
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
        return r'$\{\sigma_{f}^{2}$: ' + str(sig_var) + r', $\gamma$: ' + str(lengthscale) + r', $\sigma_{n}^{2}$: ' + str(noise_var) + '}'

    elif kernel_type == 'PER':
        
        period = hyper_params[0]
        lengthscale = hyper_params[1]
        return r'$\{p$: ' + str(period) + r', $\gamma$: ' + str(lengthscale) + '}'
        
    elif kernel_type == 'MATERN32':
        
        lengthscale = hyper_params[0]
        return r'$\gamma$: ' + str(lengthscale)  + '}'
        
    elif kernel_type == 'RQ':
        
        alpha = hyper_params[0]
        lengthscale = hyper_params[0]
        return r'$\{\alpha$: ' + str(alpha) + r', $\gamma$: ' + str(lengthscale) + '}'
    

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
    v = np.linalg.solve(L, K_s.eval())
    post_mean = np.dot(K_s.eval().T, alpha)
    #post_cov = K_ss.eval() - K_s.eval().T.dot(K_inv.eval()).dot(K_s.eval())
    post_cov = K_ss.eval() - v.T.dot(v)
    post_std = np.sqrt(np.diag(post_cov))
    return post_mean, post_cov, post_std
    
def posterior_predictive_samples(post_mean, post_cov):
    
    return np.random.multivariate_normal(post_mean, post_cov, 20)

# Metrics for assessing fit in Regression 

def rmse(post_mean, f_star):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - f_star))),3)

def log_predictive_density(predictive_density):

      return np.round(np.sum(np.log(predictive_density)), 3)

def log_predictive_mixture_density(f_star, list_means, list_cov):
      
      components = []
      for i in np.arange(len(list_means)):
            components.append(st.multivariate_normal.pdf(f_star, list_means[i].eval(), list_cov[i].eval(), allow_singular=True))
      return np.round(np.sum(np.log(np.mean(components))),3)

#-------------Plotting----------------------------------------------------

def plot_noisy_data(X, y, X_star, f_star, title):

    plt.figure()
    plt.plot(X_star, f_star, "dodgerblue", lw=3, label="True f")
    plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
    plt.xlabel("X") 
    plt.ylabel("The true f(x)") 
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
          plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
    plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
    plt.plot(X, y, 'ok', ms=3, alpha=0.5)
    plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
    plt.fill_between(np.ravel(X_star), post_mean - 1.96*post_std, 
                     post_mean + 1.96*post_std, alpha=0.2, color='g',
                     label='95% CR')
    plt.legend(fontsize='x-small')
    plt.title(title, fontsize='x-small')
    
    
def plot_lml_surface_3way(gpr, sig_var, lengthscale, noise_var):
    
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
    
def persist_datasets(X,X_star, f, f_star, y):
      
     X.tofile('X.csv', sep=',')
     X_star.tofile('X_star.csv', sep=',')
     y.tofile('y.csv', sep=',')
     f_star.tofile('f_star.csv', sep=',')

def load_datasets():
      
      X = np.asarray(pd.read_csv('X.csv', header=None))
      y = np.asarray(pd.read_csv('y.csv', header=None))
      X_star = np.asarray(pd.read_csv('X_star.csv', header=None))
      f_star = np.asarray(pd.read_csv('f_star.csv', header=None))
      return X.reshape(len(X[0]),1), y.reshape(len(y[0]),), X_star.reshape(len(X_star[0]),1), f_star.reshape(len(f_star[0]),)
      

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
    
    sig_var_true = 10.0
    lengthscale_true = 1.0
    noise_var_true = 1.0
    hyp = [sig_var_true, lengthscale_true, noise_var_true]
    cov = get_kernel('SE', [sig_var_true, lengthscale_true])
    hyp_string = get_kernel_hyp_string('SE', [sig_var_true, lengthscale_true, noise_var_true])
    
    f_all = generate_gp_latent(X_all, mean, cov)
    
    uniform = True
    X, y, X_star, f_star, f,  train_index = generate_gp_training(X_all, f_all, n_train, noise_var_true, uniform)
    #X, y, X_star, f_star = load_datasets()
    K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, noise_var_true)
    
    
    print(np.sqrt(noise_var_true))
    print(np.std(y-f))
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
    title = 'GPR' + '\n' + kernel + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_)
    plot_gp(X_star, f_star, X, y, post_mean, post_std, [], title)
    plot_kernel_matrix(post_cov.eval(),'')
    
    #---------------------------------------------------------------------
    
    # Type II ML for hyp.
    
    #---------------------------------------------------------------------
    
    kernel = Ck(10.0, (1e-10, 1e2)) * RBF(2, length_scale_bounds=(0.5, 8)) + WhiteKernel(10.0, noise_level_bounds=(1e-5,100))
    
    #kernel = Ck(constant_value=4.698, constant_value_bounds=(1e-10, 1e2)) * RBF(length_scale=2.867, length_scale_bounds=(0.5, 8)) + WhiteKernel(noise_level=0.841, noise_level_bounds=(1e-5,100))
    #kernel = Ck(4.698, (1e-10, 1e2)) * RQ(length_scale = 2.867, alpha=2.0, length_scale_bounds=(0.5, 8), alpha_bounds= (1e-5, 1e4)) + WhiteKernel(noise_level=0.841, noise_level_bounds=(1e-5,100))
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
    # Fit to data 
    gpr.fit(X, y)        
    post_mean, post_cov = gpr.predict(X_star, return_cov = True) 
    post_std = np.sqrt(np.diag(post_cov))
    post_samples = np.random.multivariate_normal(post_mean, post_cov, 10)
    rmse_ = rmse(post_mean, f_star)
    lpd_ = log_predictive_density(st.multivariate_normal.pdf(f_star, post_mean, post_cov, allow_singular=True))
    title = 'GPR' + '\n' + str(gpr.kernel_) + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_)     
    ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
    ml_deltas_dict = {'lengthscale': ml_deltas[1], 'noise_var': ml_deltas[2], 'sig_var': ml_deltas[0]}
    plot_lml_surface_3way(gpr, ml_deltas_dict['sig_var'], ml_deltas_dict['lengthscale'], ml_deltas_dict['noise_var'])

    plot_gp(X_star, f_star, X, y, post_mean, post_std, [], title)


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
post_pred_std = np.sqrt(np.diag(post_pred_cov_nf))

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
       log_l = pm.Uniform('log_l', lower=-3, upper=2)
       lengthscale = pm.Deterministic('lengthscale', tt.exp(log_l)par)
       
        #prior on noise variance
       log_nv = pm.Uniform('log_nv', lower=-5, upper=5)
       noise_var = pm.Deterministic('noise_var', tt.exp(log_nv))
         
       #prior on signal variance
       log_sv = pm.Uniform('log_sv', lower=-10, upper=5)
       sig_var = pm.Deterministic('sig_var', tt.exp(log_sv))
         
       # Specify the covariance function.
       cov_func = sig_var*pm.gp.cov.ExpQuad(1, ls=lengthscale)
    
       gp = pm.gp.Marginal(cov_func=cov_func)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
       
       # HMC Nuts auto-tuning implementation
       trace = pm.sample(draws=500, tune=1000, chains=4, discard_tuned_samples=True)
       
       map_est = pm.find_MAP()
       
       fit = pm.fit(method='fullrank_advi')
              
with hmc_gp_model:
       y_pred = gp.conditional("y_pred", X_star)
       y_trace = pm.sample_posterior_predictive(trace, samples=100)
       
# Box standard Traceplot on log axis with deltas and means highlighted
       
def get_trace_means(trace, varnames):
      
      trace_means = []
      for i in varnames:
            trace_means.append(trace[i].mean())
      return trace_means


varnames = ['sig_var', 'lengthscale','noise_var']

def trace_report(trace, varnames, priors, ml_deltas, hyp_map):
      
      ml_deltas_dict = {lengthscale: ml_deltas[1], noise_var: ml_deltas[2], sig_var: ml_deltas[0]}
      hyp_map = np.round(get_trace_means(trace, varnames),3)
      #priors = [lengthscale.distribution, noise_var.distribution, sig_var.distribution]
      #priors = [lengthscale.distribution, noise_var.distribution, sig_var.distribution]
      priors = [log_l.distribution, log_nv.distribution, log_sv.distribution]
      traces = pm.traceplot(trace, varnames=[lengthscale, noise_var, sig_var], priors=priors, prior_style='--', lines=ml_deltas_dict, bw=2, combined=True)
      traces[0][0].axvline(x=hyp_map[1], color='b',alpha=0.5, label='HMC ' + str(hyp_map[1]))
      traces[1][0].axvline(x=hyp_map[2], color='b', alpha=0.5, label='HMC ' + str(hyp_map[2]))
      traces[2][0].axvline(x=hyp_map[0], color='b', alpha=0.5, label='HMC ' + str(hyp_map[0]))
      traces[0][0].axvline(x=ml_deltas[1], color='r',alpha=0.5, label='ML ' + str(ml_deltas[1]))
      traces[1][0].axvline(x=ml_deltas[2], color='r', alpha=0.5, label='ML ' + str(ml_deltas[2]))
      traces[2][0].axvline(x=ml_deltas[0], color='r', alpha=0.5, label='ML ' + str(ml_deltas[0]))
      traces[0][0].axvline(x=lengthscale_true, color='g',alpha=0.5, label='True ' + str(lengthscale_true))
      traces[1][0].axvline(x=noise_var_true, color='g', alpha=0.5, label= 'True ' + str(noise_var_true))
      traces[2][0].axvline(x=sig_var_true, color='g', alpha=0.5, label='True ' + str(sig_var_true))
      traces[0][1].axhline(y=hyp_map[1], color='b', alpha=0.5)
      traces[1][1].axhline(y=hyp_map[2], color='b', alpha=0.5)
      traces[2][1].axhline(y=hyp_map[0], color='b', alpha=0.5)
      traces[0][0].axes.set_xscale('log')
      traces[1][0].axes.set_xscale('log')
      traces[2][0].axes.set_xscale('log')
      traces[0][0].axes.set_xticks([0.1,1,10])
      traces[1][0].axes.set_xticks([0.01,0.1,1,10])
      traces[2][0].axes.set_xticks([0.1,1,10,100])
      traces[0][0].hist(trace['lengthscale'], bins=100, normed=True, color='orange', alpha=0.3)
      traces[1][0].hist(trace['noise_var'], bins=100, normed=True, color='orange', alpha=0.3)
      traces[2][0].hist(trace['sig_var'], bins=100, normed=True, color='orange', alpha=0.3)
      for j, k in [(0,0), (1,0), (2,0)]:
            traces[j][k].legend(fontsize='x-small')

# Write out trace summary & autocorrelation plots

prefix = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Config/'
summary_df = pm.summary(trace)
summary_df['Acc Rate'] = np.mean(trace.get_sampler_stats('mean_tree_accept'))
np.round(summary_df,3).to_csv(prefix + 'trace_summary_se_unif_25.csv')
pm.autocorrplot(trace, varnames=['sig_var', 'noise_var', 'lengthscale'], burn=100)

# Compute posterior predictive mean and covariance - careful (not so obvious)

def get_combined_trace(trace):
      
    trace_df = pd.DataFrame()
    trace_df['sig_var'] = np.mean(trace.get_values('sig_var', combine=False), axis=0)
    trace_df['lengthscale'] = np.mean(trace.get_values('lengthscale', combine=False), axis=0)
    trace_df['noise_var'] = np.mean(trace.get_values('noise_var', combine=False), axis=0)
    return trace_df

def get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv, y):
      
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

def get_post_mcmc_mean_cov(trace_df, X, X_star, y, varnames, n_train, test_size):
      
      list_means = []
      list_cov = []
      list_mean_sq = []
      for i in range(0,len(trace_df), 10):
            print(i)
            theta = get_joint_value_from_trace(trace_df, varnames, i) 
            cov = get_kernel('SE', [theta[0], theta[1]])
            K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, theta[2])
            mean_i = get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv, y)
            cov_i = get_post_cov_theta(theta, K_s, K_ss, K_inv)
            list_means.append(mean_i)
            list_cov.append(cov_i)
            list_mean_sq.append(tt.outer(mean_i, mean_i))
      post_mean_trace = tt.mean(list_means, axis=0)
      post_mean_trace = post_mean_trace.eval()
      print('Mean evaluated')
      post_cov_mean =  tt.mean(list_cov, axis=0) 
      post_cov_mean = post_cov_mean.eval() 
      print('Cov evaluated')
      outer_means = tt.mean(list_mean_sq, axis=0)
      outer_means = outer_means.eval()
      print('SSQ evaluated')
      post_cov_trace = post_cov_mean + outer_means - np.outer(post_mean_trace, post_mean_trace)
      return post_mean_trace, post_cov_trace, post_cov_mean, list_means, list_cov
     
varnames = ['sig_var', 'lengthscale','noise_var']
trace_df = get_combined_trace(trace)
theta = get_trace_means(trace_df, varnames=['sig_var', 'lengthscale','noise_var'])
test_size = len(X_all) - len(X)
post_mean_trace, post_cov_trace, post_cov_mean, list_means, list_cov = get_post_mcmc_mean_cov(trace_df, X, X_star,y, varnames, n_train, test_size)
post_std_trace = np.sqrt(np.diag(post_cov_trace))
post_std_mean = np.sqrt(np.diag(post_cov_mean))

means_df = np.empty((180,))
for i in range(0,50):
      print(i)
      means_df = np.column_stack((means_df, list_means[i].eval()))      
means_df = np.delete(means_df, 0,1)

std_df = np.empty((180,))
for i in range(0,50):
      print(i)
      std_df = np.column_stack((std_df, np.diag(list_cov[i].eval())))      
std_df = np.delete(std_df, 0,1)

post_std_of_means = np.std(means_df, 0, 1)

# Fit metrics - All 3 cases : ML, HMC, MAP

rmse_hmc = rmse(f_star, post_mean_trace)
rmse_ml = rmse(f_star, post_pred_mean)

lpd_hmc = log_predictive_mixture_density(f_star, list_means, list_cov)
lpd_ml = log_predictive_density(st.multivariate_normal.pdf(f_star, post_pred_mean, post_pred_cov, allow_singular=True))

# Type II vs. HMC Report 

plt.figure(figsize=(12,7))

# Plot fit with mean and variance ML case

plt.subplot(221)
#post_samples = posterior_predictive_samples(post_pred_mean, post_pred_cov_nf)
#plt.plot(X_star, post_samples.T, color='grey', alpha=0.05)
plt.fill_between(np.ravel(X_star), post_pred_mean - 2*post_pred_std, 
                     post_pred_mean + 2*post_pred_std, alpha=0.2, color='r',
                     label=r'$2\sigma_{*}^{2(II ML)}$')
plt.plot(X_star, f_star, "black", lw=1.0, linestyle='dashed', label="True f",alpha=0.5);
plt.plot(X_star, post_pred_mean, color='r', label=r'$\mu_{*}^{II ML}$')
plt.plot(X, y, 'ok', ms=3, alpha=0.5)
plt.legend(fontsize='small')
plt.title('Type II ML' + '\n' 
          'True: ' + get_kernel_hyp_string('SE', [sig_var_true, lengthscale_true, noise_var_true])  + '\n' +
          'ML: ' + get_kernel_hyp_string('SE', np.round(ml_deltas, 3)), fontsize='medium')
plt.ylim(-9,9)

# Plot fit with mean and variance HMC case

plt.subplot(222)
#post_samples = posterior_predictive_samples(post_mean_trace, post_cov_trace)
#plt.plot(X_star, post_samples.T, color='grey', alpha=0.1)
plt.fill_between(np.ravel(X_star), (post_mean_trace - 2*post_std_trace).reshape(test_size,), 
                     (post_mean_trace + 2*post_std_trace).reshape(test_size,), alpha=0.2, color='b',
                     label=r'$2\sigma_{*}^{2 (HMC)}$')
plt.plot(X_star, f_star, "black", lw=1.0, linestyle='dashed', label="True f",alpha=0.5);
plt.plot(X_star, post_mean_trace, color='b', label=r'$\mu_{*}^{HMC}$')
plt.plot(X, y, 'ok', ms=3, alpha=0.5)
plt.legend(fontsize='x-small')
plt.title('HMC' + '\n' 
          'True: ' + get_kernel_hyp_string('SE', [sig_var_true, lengthscale_true, noise_var_true])  + '\n' +
          'HMC: ' + get_kernel_hyp_string('SE', np.round(theta, 3))
          , fontsize='medium')
plt.ylim(-9,9)

# Plot the space of means and uncertainties overlaid

plt.subplot(223)
for i in range(0,50):
      plt.fill_between(np.ravel(X_star), means_df[:,i] - 2*std_df[:,i],  means_df[:,i] + 2*std_df[:,i], 
                       alpha=0.3, color='grey')
plt.plot(X_star, means_df, alpha=0.3)
plt.plot(X_star, f_star, "black", lw=2.0, linestyle='dashed', label="True f",alpha=0.5);
plt.plot(X, y, 'ok', ms=3)
plt.title('Posterior means and variances per ' + r'$\theta_{i}$')
plt.ylim(-9,9)

# Plot overlays 

plt.subplot(224)
plt.fill_between(np.ravel(X_star), (post_mean_trace - 2*post_std_trace).reshape(test_size,), 
                     (post_mean_trace + 2*post_std_trace).reshape(test_size,), alpha=0.15, color='b',
                     label=r'HMC')
plt.fill_between(np.ravel(X_star), post_pred_mean - 2*post_pred_std, 
                     post_pred_mean + 2*post_pred_std, alpha=0.15, color='r',
                     label=r'ML')
plt.plot(X_star, f_star, "black", lw=2.0, linestyle='dashed', label="True f",alpha=0.5);
plt.plot(X_star, post_pred_mean, color='r')
plt.plot(X_star, post_mean_trace, color='b')
plt.plot(X, y, 'ok', ms=3, alpha=0.5)
plt.title('Type II ML - ' + 'RMSE: ' + str(rmse_ml) + '  LPD: ' + str(lpd_ml) + '\n' + 
          'HMC        - ' + 'RMSE: ' + str(rmse_hmc) + '  LPD: ' + str(lpd_hmc), fontsize='small')
plt.legend(fontsize='x-small')
plt.ylim(-9,9)

sns.set(style="ticks")

g = sns.PairGrid(trace_df, palette="Set2")
g = g.map_upper(plt.scatter, s=0.5)
g = g.map_lower(sns.kdeplot, cmap="Blues_d", n_levels=20)
g = g.map_diag(sns.kdeplot, lw=1, legend=False)






with pm.Model() as model:
      
      x = pm.MvNormal('x',[0,0], cov=np.array([[1,0.8],[0.8,1]]), shape=(2,2))
      trace2 = pm.sample()



## Plot fit with mean and variance MAP case

# MAP Case

#K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(cov, X, X_star, n_train, theta[2])
#post_mean_map = get_post_mean_theta(theta, K, K_s, K_ss, K_noise, K_inv).eval()
#post_cov_map = get_post_cov_theta(theta, K_s, K_ss, K_inv).eval()
#post_std_map = np.sqrt(np.diag(post_cov_map))
#
#rmse_map = rmse(f_star, post_mean_map)
#lpd_map = log_predictive_density(st.multivariate_normal.pdf(f_star, post_mean_map, post_cov_map, allow_singular=True))

#
#plt.figure()
#plt.plot(X_star, post_mean_map, color='g', label=r'$\mu_{*}^{MAP}$')
#plt.fill_between(np.ravel(X_star), (post_mean_map - 2*post_std_map).reshape(test_size,), 
#                     (post_mean_map + 2*post_std_map).reshape(test_size,), alpha=0.2, color='g',
#                     label=r'$2\sigma_{*}^{2 (MAP)}$')


