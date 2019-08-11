#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 15 15:12:29 2019

@author: vidhi

Generate data from an ARD kernel of custom dimension

"""
from mpl_toolkits.mplot3d import Axes3D 

# Generate data from a 5d ARD kernel 

def generate_gp_latent(X_all, mean, cov):
      
      return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval())

if __name__ == "__main__":

    d = 2
    
    n_train = 20
    n_star = 200
    
    xmin = 0
    xmax = 30
        
    #X = np.random.uniform(low=xmin, high=xmax, size=(100,d))
    
    x_ = np.linspace(1, 20, 10)
    X1, X2 = np.meshgrid(x_, x_)

    X_all = np.vstack([X1.ravel(), X2.ravel()]).T
    
    plt.plot(X_all, 'ko')

    # A mean function that is zero everywhere
    
    mean = pm.gp.mean.Zero()
    
    # Kernel Hyperparameters 
    
    sig_sd_true = 10.0
    lengthscale_true = [3.0 ,5.0]
    noise_sd = np.sqrt(50)
    
    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]
    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(2, lengthscale_true)
    
    f_all = generate_gp_latent(X_all, mean, cov)
    
    plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.contourf(X1, X2,  Z=f_all.reshape(10,10))