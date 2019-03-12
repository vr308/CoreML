#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:05:01 2019

@author: vidhi

"""
import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import scipy.stats as st
import theano.tensor as tt
from sklearn.gaussian_process import GaussianProcessRegressor

from kernel import Kernel
from data_scheme import GP_Synthetic_Data

np.set_printoptions(precision=3)

#TODO: Encode Prior logic

class GP: 
      
      def __init__(self, input_dim, data, mean, kernel):
            
            self.input_dim = input_dim
            self.data = data
            self.mean = mean
            self.kernel = kernel
            self.ml_deltas = None
            
      def inf_type_II_ml(self):
            
            X = self.data.X
            y = self.data.y
            f_star = self.data.f_star
            X_star = self.data.X_star
            
            kernel = self.kernel.sklearn_kernel
            gpr = GaussianProcessRegressor(kernel=kernel)
                    
            # Fit to data 
            gpr.fit(X, y)        
            post_mean, post_cov = gpr.predict(X_star, return_cov = True) 
            post_std = np.sqrt(np.diag(post_cov))
            ml_deltas = np.round(np.exp(gpr.kernel_.theta), 3)
            hyperparam_names = list(self.kernel.hyperparams.keys())
            hyperparam_names.append('noise_var')
            self.ml_deltas = dict(zip(hyperparam_names, ml_deltas))  
            return post_mean, post_cov, post_std
            
      def get_ml_deltas_hyp_string():
            
      def inf_analytical(self, kernel, pred_noise):
            
            K, K_s, K_ss, K_noise, K_inv = kernel.get_kernel_matrix_blocks(self.data.X, self.data.X_star)
            
            L = np.linalg.cholesky(K_noise.eval())
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.data.y))
            v = np.linalg.solve(L, K_s.eval())
            post_mean = np.dot(K_s.eval().T, alpha)
            #post_cov = K_ss.eval() - K_s.eval().T.dot(K_inv.eval()).dot(K_s.eval())
            if pred_noise:
                n_star = len(gp.data.X_star)
                post_cov = K_ss.eval() - v.T.dot(v) + self.kernel.noise_var*tt.eye(n_star).eval()
            else:
                post_cov = K_ss.eval() - v.T.dot(v)
            post_std = np.sqrt(np.diag(post_cov))
            return post_mean, post_cov, post_std
            
      def inf_hmc(self):
            
            with pm.Model() as hmc_gp_model:
        
                   # prior on lengthscale 
                   log_l = pm.Uniform('log_l', lower=-3, upper=3)
                   l = pm.Deterministic('l', tt.exp(log_l))
                     
                   #prior on signal variance
                   log_sv = pm.Uniform('log_sv', lower=-10, upper=5)
                   sig_var = pm.Deterministic('sig_var', tt.exp(log_sv))
                     
                   #prior on noise variance
                   log_nv = pm.Uniform('log_nv', lower=-10, upper=5)
                   noise_var = pm.Deterministic('noise_var', tt.exp(log_nv))
                   
                   # Specify the covariance function.
                   cov_func = self.kernel.kernel_func
                
                   gp = pm.gp.Marginal(cov_func=cov_func)
                        
                   # Marginal Likelihood
                   y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
                        
                   hmc_step = pm.step_methods.HamiltonianMC(path_length=10.0, target_accept=0.651)
                   
                   # HMC Nuts auto-tuning implementation
                   trace = pm.sample(draws=500, step=hmc_step)
                          
            with hmc_gp_model:
                   y_pred = gp.conditional("y_pred", X_star)
             
            return trace
            
      def inf_variational(self):
            
            return
            #TODO
            
      def inf_laplace(self):
            
            return
            #TODO
      
      @staticmethod
      def posterior_predictive_samples(post_mean, post_cov):
    
            return np.random.multivariate_normal(post_mean, post_cov, 20)

      # Metrics for assessing fit in Regression 

      def rmse(self, post_mean):
    
            return np.round(np.sqrt(np.mean(np.square(post_mean - self.data.f_star))),3)

      def log_predictive_density(self, predictive_density):

            return np.round(np.sum(np.log(predictive_density)), 3)

      def log_predictive_mixture_density(self, list_means, list_cov):
      
            components = []
            for i in np.arange(len(list_means)):
                  components.append(st.multivariate_normal.pdf(self.data.f_star, list_means[i].eval(), list_cov[i].eval(), allow_singular=True))
            return np.round(np.sum(np.log(np.mean(components))),3)
      
      # Plotting         
      def plot_gp(self, post_mean, post_std, post_samples, title, new_fig=True):
    
          X_star = self.data.X_star
          X = self.data.X
          y = self.data.y
          f_star = self.data.f_star
          
          if new_fig:
                plt.figure()
          if post_samples != []:
                plt.plot(X_star, post_samples.T, color='grey', alpha=0.2)
          plt.plot(X_star, f_star, "dodgerblue", lw=1.4, label="True f",alpha=0.7);
          plt.plot(X, y, 'ok', ms=3, alpha=0.5)
          plt.plot(X_star, post_mean, color='r', lw=2, label='Posterior mean')
          plt.fill_between(np.ravel(X_star), post_mean - 2*post_std, 
                           post_mean + 2*post_std, alpha=0.2, color='r',
                           label=r'2$\sigma$')
          plt.legend(fontsize='small')
          plt.title(title, fontsize='x-small')
         
      def plot_type_II_hmc_report():
            
         return
          
if __name__ == "__main__":

        # Set Data Scheme
        
        print('-----------Generating Synthetic Data to Fit-----------------------')
      
        mean = pm.gp.mean.Zero()
        
        kernel = Kernel('SE', 1, {'sig_var': 2, 'lengthscale':1.0}, noise_var=0.4)
        
        data = GP_Synthetic_Data(0,13,200,20, mean, kernel, uniform=True)
        data.plot_y('Training data (iid noise)')
        
        # GP Instance
        
        print('------------Instantiating object of GP class------------------------')
      
        gp = GP(1, data, mean, kernel)
        
        X = gp.data.X
        X_star = gp.data.X_star
        y = gp.data.y
        
        print('Type II ML Results')
        
        post_mean, post_cov, post_std = gp.inf_type_II_ml()
        post_samples = GP.posterior_predictive_samples(post_mean, post_cov)
        
        #kernel_al = Kernel('RQ', 1, gp.ml_deltas, noise_var=gp.ml_deltas['noise_var'])
        #post_mean_al, post_cov_al, post_std_al = gp.inf_analytical(kernel_al, True)
        
        rmse_ml = gp.rmse(post_mean)
        lpd_ml = gp.log_predictive_density(st.multivariate_normal.pdf(gp.data.f_star, post_mean, post_cov, allow_singular=True))
        title = 'GPR' + '\n' + kernel.get_kernel_hyp_string() + '\n' + 'RMSE: ' + str(rmse_ml) + '\n' + 'LPD: ' + str(lpd_ml)     
        
        gp.plot_gp(post_mean, post_std, [], title, True)
         
        print('HMC Results')
        
        trace = gp.inf_hmc() 

        # Trace Results
        
        trace_hyp = Trace_Hyp_GP(trace, gp)
      
            
      