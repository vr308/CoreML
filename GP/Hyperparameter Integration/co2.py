#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:40:17 2019

@author: vidhi

"""

import numpy as np
import pymc3 as pm
import scipy.stats as st
import matplotlib.pylab as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
import theano.tensor as tt

def load_co2_data():
      
      df = pd.read_table('mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
      return df.dropna(axis=0)

      
if __name__ == "__main__":

      df = load_co2_data()
      
      # PYMC3 
      
      # se +  sexper + rq + se + noise
      
      sig_var_1 = 10
      lengthscale_2 = 1 
      sig_var_3 =10
      lengthscale_4 = 1 
      lengthscale_5 = 1
      sig_var_6 = 10
      lengthscale_7 = 1 
      alpha_8 = 1.0
      sig_var_9 = 10
      lengthscale_10 = 1.0 
      noise_11 = 0.5
      
      k1 = pm.gp.cov.Constant(sig_var_1)*pm.gp.cov.ExpQuad(1, lengthscale_2) 
      k2 = pm.gp.cov.Constant(sig_var_3)*pm.gp.cov.ExpQuad(1, lengthscale_4)*pm.gp.cov.Periodic(1, period=1, ls=lengthscale_5)
      k3 = pm.gp.cov.Constant(sig_var_6)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=lengthscale_7)
      k4 = pm.gp.cov.Constant(sig_var_9)*pm.gp.cov.ExpQuad(1, lengthscale_10) +  pm.gp.cov.WhiteNoise(noise_11)
      
      k = k1 + k2 + k3 +k4
      
      # sklearn kernel 

      sk1 = Ck(sig_var_1)*RBF(lengthscale_2)
      sk2 = Ck(sig_var_3)*RBF(lengthscale_4)*PER(length_scale=lengthscale_5, periodicity=1, periodicity_bounds='fixed')
      sk3 = Ck(sig_var_6)*RQ(length_scale= lengthscale_7, alpha=alpha_8) 
      sk4 = Ck(sig_var_9)*RBF(lengthscale_10) + WhiteKernel(noise_11)
      
      #---------------------------------------------------------------------
    
      # Type II ML for hyp.
    
      #---------------------------------------------------------------------
    
     sk_kernel = sk1 + sk2 + sk3 + sk4
    
     gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=10, normalize_y = True)
        
     # Fit to data 
     
     X = np.array(df.year).reshape(len(df),1)
     y = df.co2 
     
     gpr.fit(X, y)
     
     print("\nLearned kernel: %s" % gpr.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gpr.log_marginal_likelihood(gpr.kernel_.theta))

      # Prediction
     
     X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
     y_pred, y_std = gpr.predict(X_, return_std = True) 

     # Plotting 
     
     plt.scatter(X, y, c='k')
     plt.plot(X_, y_pred)
     plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
     plt.xlim(X_.min(), X_.max())
     plt.xlabel("Year")
     plt.ylabel(r"CO$_2$ in ppm")
     plt.title(r"Atmospheric CO$_2$ concentration")
     plt.tight_layout()
     
     #-----------------------------------------------------
      
     #       Hybrid Monte Carlo
          
     #-----------------------------------------------------

     with pm.Model() as hyp_learning:
        
             # prior on lengthscales
             
             log_l2 = pm.Uniform('log_l2', lower=-5, upper=6)
             log_l4 = pm.Uniform('log_l4', lower=-5, upper=6)
             log_l5 = pm.Uniform('log_l5', lower=-5, upper=6)
             log_l7 = pm.Uniform('log_l7', lower=-5, upper=6)
             log_l10 = pm.Uniform('log_l10', lower=-5, upper=6)

             l2 = pm.Deterministic('l2', tt.exp(log_l2))
             l4 = pm.Deterministic('l4', tt.exp(log_l4))
             l5 = pm.Deterministic('l5', tt.exp(log_l5))
             l7 = pm.Deterministic('l7', tt.exp(log_l7))
             l10 = pm.Deterministic('l10', tt.exp(log_l10))
             
             # prior on amplitudes
             
             log_sv1 = pm.Uniform('log_sv1', lower=-10, upper=10)
             log_sv3 = pm.Uniform('log_sv3', lower=-10, upper=10)
             log_sv6 = pm.Uniform('log_sv6', lower=-10, upper=10)
             log_sv9 = pm.Uniform('log_sv9', lower=-10, upper=10)

             sig_var1 = pm.Deterministic('sig_var1', tt.exp(log_sv1))
             sig_var3 = pm.Deterministic('sig_var3', tt.exp(log_sv3))
             sig_var6 = pm.Deterministic('sig_var6', tt.exp(log_sv6))
             sig_var9 = pm.Deterministic('sig_var9', tt.exp(log_sv9))

             # prior on alpha
            
             log_alpha8 = pm.Uniform('log_alpha8', lower=-10, upper=10)
             alpha8 = pm.Deterministic('alpha8', tt.exp(log_alpha8))
             
             # prior on noise variance term
            
             log_nv11 = pm.Uniform('log_nv11', lower=-5, upper=5)
             noise_var = pm.Deterministic('noise_var', tt.exp(log_nv11))
               
             
             # Specify the covariance function
             
             k1 = pm.gp.cov.Constant(sig_var1)*pm.gp.cov.ExpQuad(1, l2) 
             k2 = pm.gp.cov.Constant(sig_var3)*pm.gp.cov.ExpQuad(1, l4)*pm.gp.cov.Periodic(1, period=1, ls=l5)
             k3 = pm.gp.cov.Constant(sig_var6)*pm.gp.cov.RatQuad(1, alpha=alpha8, ls=l7)
             k4 = pm.gp.cov.Constant(sig_var9)*pm.gp.cov.ExpQuad(1, l10) +  pm.gp.cov.WhiteNoise(noise_var)
      
             k = k1 + k2 + k3 +k4
                
             gp = pm.gp.Marginal(cov_func=k)
                  
             # Marginal Likelihood
             y_ = gp.marginal_likelihood("y", X=X, y=y, noise=np.sqrt(noise_var))
             
             # HMC Nuts auto-tuning implementation
             
             #trace_hmc = pm.sample(draws=250,chains=2)
             
             advi = pm.FullRankADVI(n_init=50000)
             advi.fit()    
             trace_advi = advi.approx.sample(draws=50000, include_transformed=False) 
                   
      with hyp_learning:
             f_pred = gp.conditional("f_pred", X_star)
    

varnames = ['l2','l4','l5','l7','l10','sig_var1','sig_var3','sig_var6','sig_var9','alpha8','noise_var']         

def get_trace_df(trace_hmc, varnames):
      
    trace_df = pd.DataFrame()
    for i in varnames:
          trace_df[i] = trace_hmc.get_values(i)
    