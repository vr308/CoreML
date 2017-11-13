#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:28:58 2017

@author: vr308

Basic Regression Examples

Least squares

Quantities of interest:
    
    beta0, beta1
    R^2 : A unit-less quantity that summarizes the goodness of fit
    Std error of the estimate: Average error in predictions in units of the response variable
    Std error of the gradient (beta1) / Std error of the intercept (beta0)
    Confidence Intervals of gradient, intercept and mean of the response.
    Prediction Intervals: Applies to calculating the uncertainty bands for test data points.
    Estimate of the variance of the residual (sigma^2)  = SQRT(Mean squared error) = std error of the estimate
    
Single Regressor / Multiple Regressors

"""
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.optimize as so
import statsmodels.api as smt
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def analytical_multiple_leastsq(X,y):
            
        # There are multiple regressors, potentially non-linear functions of x. 
        
        n = len(X)
        Beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
        fitted = np.dot(X, Beta)
        ssq_deviations = np.sum(np.power(y - fitted,2))
        std_error = np.sqrt(ssq_deviations/(n-4))
        std_errors_coefs = std_error*np.sqrt(np.linalg.inv(np.dot(X.T,X)).diagonal())
        
        Xframe = pd.DataFrame(X)
        std_error_pred = []
        std_error_mean = []
        for index, row in Xframe.iterrows():
            row = np.asarray(row)
            factor0 = np.sqrt(1 + np.dot(row.T,np.dot(np.linalg.inv(np.dot(X.T,X)), row)))
            factor1 = np.sqrt(np.dot(row.T,np.dot(np.linalg.inv(np.dot(X.T,X)), row)))
            std_error_pred.append(std_error*factor0)
            std_error_mean.append(std_error*factor1)
        return Beta, std_error, std_errors_coefs, std_error_pred, std_error_mean

        

def analytical_simple_leastsq(x,y):     
    
        n = len(x)
        beta1 = np.divide(n*sum(x*y) - sum(x)*sum(y),n*sum(x*x) - np.power(sum(x), 2))
        beta0 = np.mean(y) - beta1*np.mean(x)
        fitted = beta0 + beta1*x
        
        ssq_deviations = np.sum(np.power(y - fitted,2))
        std_error = np.sqrt(ssq_deviations/(n-2))
            
        ssq_x = np.sum(np.power((x - np.mean(x)), 2))
        std_error_slope = std_error/np.sqrt(ssq_x)
        std_error_int = std_error*np.sqrt(np.sum(x**2)/(n*ssq_x))
        std_error_pred = std_error*np.sqrt(1 + 1.0/n + ((x - np.mean(x))**2)/ssq_x)
        std_error_mean = std_error*np.sqrt(1.0/n + ((x - np.mean(x))**2)/ssq_x)

        #Confidence Intervals
        
        # Slope
        upper_slope = beta1 +  st.t.ppf(0.975, 98)*std_error_slope
        lower_slope = beta1 -  st.t.ppf(0.975, 98)*std_error_slope
        
        # Intercept 
        upper_int = beta0 +  st.t.ppf(0.975, 98)*std_error_int
        lower_int = beta0 -  st.t.ppf(0.975, 98)*std_error_int
        
        # Mean
        upper_mean = fitted +  st.t.ppf(0.975, 98)*std_error_mean
        lower_mean = fitted -  st.t.ppf(0.975, 98)*std_error_mean
        
        # Prediction Interval
        upper_pred = fitted + st.t.ppf(0.975,98)*std_error_pred
        lower_pred = fitted - st.t.ppf(0.975,98)*std_error_pred
        
        return beta1, beta0, std_error, std_error_slope, (upper_slope, lower_slope), (upper_int, lower_int), (upper_mean, lower_mean), (upper_pred, lower_pred)

if __name__ == "__main__":
    
    x = np.linspace(1,20,100)
    y_true = 0.5*np.sin(x) + 0.5*x + -0.02*(x-5)**2
    y_noise = y_true + 0.2*np.std(y_true)*np.random.normal(0,1,100)
    
    # Fit a line ax + b - least squares [3 methods]
    slope, intercept, r_value, p_value, std_err = st.linregress(x,y_noise)
    slope, intercept = np.linalg.lstsq(np.vstack([x,np.ones(len(x))]).T,y_noise)[0]
    
    X = smt.add_constant(x)
    model = smt.OLS(y_noise, X)
    results = model.fit()
    
    sdev, pred_lower, pred_upper = wls_prediction_std(results, alpha=0.05)
    
    # Verification
    
    slope, intercept, std_error, std_error_slope, CI_slope, CI_int, CI_mean,  PI =  analytical_simple_leastsq(x,y_noise)
     
    #Visualize the data
    
    plt.figure()
    plt.plot(x,y_true, 'b', label='True Function')
    plt.scatter(x,y_noise, s=4)
    plt.plot(x, x*slope + intercept, label = 'Fitted')
    plt.fill_between(x, x*CI_slope[0] + CI_int[0], x*CI_slope[1] + CI_int[1], color= 'b', alpha=0.4, label = 'CI for the coefs')
    plt.fill_between(x, PI[0], PI[1], color= 'r', alpha=0.1, label= 'PI')
    plt.fill_between(x, CI_mean[0], CI_mean[1], color= 'g', alpha=0.1, label = 'CI for E(y)')
    plt.title('Fitting a Linear Line')
    plt.legend()
    
    #------------------------------------------------------------------------------------------------------------------------
    
    # Fit a non-linear curve to data (linear in parameters) - least squares
    
    X = np.vstack([x**3, x**2, x, np.ones(len(x))]).T
    coef0, coef1, coef2, intercept = np.linalg.lstsq(X,y_noise)[0]
    results = smt.OLS(y_noise, X).fit()
    results.summary()
    
    sdev, pred_lower, pred_upper = wls_prediction_std(results, alpha=0.05)
    
    # Standard errors of the co-efficients
    results.bse
    
    # Standard errors of the regression
    np.sqrt(results.mse_resid)
    
    # Verification
    
    Beta, std_error, std_errors_coefs, std_error_pred, std_error_mean = analytical_multiple_leastsq(X,y_noise)
    fitted = coef0*(x**3) + coef1*(x**2) + coef2*(x) + intercept
    
    # Confidence Intervals

    CI_lower_beta = Beta - st.t.ppf(0.975,96)*std_errors_coefs
    CI_upper_beta = Beta + st.t.ppf(0.975,96)*std_errors_coefs
    lower_CI_limit = (x**3)*CI_lower_beta[0] + (x**2)*CI_lower_beta[1] + x*CI_lower_beta[2] + CI_lower_beta[3]
    upper_CI_limit = (x**3)*CI_upper_beta[0] + (x**2)*CI_upper_beta[1] + x*CI_upper_beta[2] + CI_upper_beta[3]   
    
    CI_lower_mean = fitted - st.t.ppf(0.975,96)*np.asarray(std_error_mean)
    CI_upper_mean = fitted + st.t.ppf(0.975,96)*np.asarray(std_error_mean)

    #Prediction Interval 
    
    PI_lower = fitted - st.t.ppf(0.975,96)*np.asarray(std_error_pred) 
    PI_upper = fitted + st.t.ppf(0.975,96)*np.asarray(std_error_pred)
  
    #Visualize the data
    
    plt.figure()
    plt.plot(x,y_true, 'b')
    plt.scatter(x,y_noise, s=4)
    plt.plot(x, fitted)
    #plt.fill_between(x, lower_CI_limit, upper_CI_limit, color= 'b', alpha=0.4)
    plt.fill_between(x, pred_lower, pred_upper, color= 'b', alpha=0.1)
 #   plt.fill_between(x, PI_lower, PI_upper, color= 'r', alpha=0.1)
    plt.fill_between(x, CI_lower_mean, CI_upper_mean, color= 'g', alpha=0.1)
    plt.title('Fitting a Curve')
    
    
   


