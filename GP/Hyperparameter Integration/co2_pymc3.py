#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:29:02 2019

@author: vidhi
"""

import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pylab as plt
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Span, Label, Legend
from bokeh.io import output_notebook
from bokeh.palettes import brewer
import  scipy.stats as st 
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")


def normalize(y):
      
      return (y - y[0])/np.std(y)

path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2'
df = pd.read_table('mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)

# creat a date index for the data - convert properly from the decimal year 

#df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')

df.dropna(inplace=True)

first_co2 = df['co2'][0]
std_co2 = np.std(df['co2'])   

# normalize co2 levels
   
y = normalize(df['co2'])
t = df['year'] - df['year'][0]

sep_idx = 650

y_train = y[0:sep_idx].values
y_test = y[sep_idx:].values

t_train = t[0:sep_idx].values[:,None]
t_test = t[sep_idx:].values[:,None]

#data_monthly = pd.read_csv(pm.get_data("monthly_in_situ_co2_mlo.csv"), header=56)
#
## - replace -99.99 with NaN
#data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)
#
## fix column names
#cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
#        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]
#data_monthly.columns = cols
#cols.remove("--"); cols.remove("--")
#data_monthly = data_monthly[cols]
#
#data_monthly.dropna(inplace=True)
#
## fix time index
#data_monthly["day"] = 15
#data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
#cols.remove("year"); cols.remove("month")
#data_monthly = data_monthly[cols]
#
#data_monthly.head(5)
#
#def dates_to_idx(timelist):
#    reference_time = pd.to_datetime('1958-03-15')
#    t = (timelist - reference_time) / pd.Timedelta(1, "Y")
#    return np.asarray(t)
#
#t = dates_to_idx(data_monthly.index)
#
## normalize CO2 levels
#y = data_monthly["CO2"].values
#first_co2 = y[0]
#std_co2 = np.std(y)
#y_n = (y - first_co2) / std_co2
#
#data_monthly = data_monthly.assign(t = t)
#data_monthly = data_monthly.assign(y_n = y_n)
#
#sep_idx = data_monthly.index.searchsorted(pd.to_datetime("2003-12-15"))
#data_early = data_monthly.iloc[:sep_idx+1, :]
#data_later = data_monthly.iloc[sep_idx:, :]
#
#p = figure(x_axis_type='datetime', title='Monthly CO2 Readings from Mauna Loa',
#           plot_width=550, plot_height=350)
#p.yaxis.axis_label = 'CO2 [ppm]'
#p.xaxis.axis_label = 'Date'
#predict_region = BoxAnnotation(left=pd.to_datetime("2003-12-15"),
#                               fill_alpha=0.1, fill_color="firebrick")
#p.add_layout(predict_region)
#ppm400 = Span(location=400,
#              dimension='width', line_color='red',
#              line_dash='dashed', line_width=2)
#p.add_layout(ppm400)
#
#p.line(data_monthly.index, data_monthly['CO2'],
#       line_width=2, line_color="black", alpha=0.5)
#p.circle(data_monthly.index, data_monthly['CO2'],
#         line_color="black", alpha=0.1, size=2)
#
#train_label = Label(x=100, y=165, x_units='screen', y_units='screen',
#                    text='Training Set', render_mode='css', border_line_alpha=0.0,
#                    background_fill_alpha=0.0)
#test_label  = Label(x=585, y=80, x_units='screen', y_units='screen',
#                    text='Test Set', render_mode='css', border_line_alpha=0.0,
#                    background_fill_alpha=0.0)

#p.add_layout(train_label)
#p.add_layout(test_label)
#show(p)
#
## pull out normalized data
#t = data_early["t"].values[:,None]
#y = data_early["y_n"].values


# sklearn kernel 

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

# LT trend
sk1 = Ck(sig_var_1)*RBF(lengthscale_2)

# Periodic x LT Trend
sk2 = Ck(sig_var_3)*RBF(lengthscale_4)*PER(length_scale=lengthscale_5, periodicity=1, periodicity_bounds='fixed')

# Medium term irregularities
sk3 = Ck(sig_var_6)*RQ(length_scale= lengthscale_7, alpha=alpha_8) 

# Noise model
sk4 = Ck(sig_var_9)*RBF(lengthscale_10) + WhiteKernel(noise_11)

#---------------------------------------------------------------------
    
# Type II ML for hyp.
    
#---------------------------------------------------------------------
    
sk_kernel = sk1 + sk2 + sk3 + sk4
gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=10)

# Fit to data 

gpr.fit(t_train, y_train)
     
print("\nLearned kernel: %s" % gpr.kernel_)
print("Log-marginal-likelihood: %.3f"
% gpr.log_marginal_likelihood(gpr.kernel_.theta))


print("Predicting with trained gp on training data")

mu_fit_n, std_fit_n = gpr.predict(t_train, return_std=True)
mu_fit = mu_fit_n*std_co2 + first_co2
std_fit = std_fit_n*std_co2

print("Predicting with trained gp on test data")

mu_test_n, cov_test_n = gpr.predict(t_test, return_cov=True)
mu_test = mu_test_n*std_co2 + first_co2
std_test  = np.sqrt(np.diag(cov_test_n))*std_co2

rmse_ = np.round(np.sqrt(np.mean(np.square(mu_test - (y_test*std_co2 + first_co2)))), 2)

lpd = lambda x : st.norm.pdf(x, mu_test_n[0], std_test[0])
lpd_ = np.round(np.sum(lpd(y_test)),2) 

plt.figure()
plt.plot(df['year'], df['co2'], 'ko', markersize=1)
plt.plot(df['year'][0:sep_idx], mu_fit, alpha=0.5, label='y_pred_train', color='b')
plt.plot(df['year'][sep_idx:], mu_test, alpha=0.5, label='y_pred_test', color='r')
plt.fill_between(df['year'][0:sep_idx], mu_fit - 2*std_fit, mu_fit + 2*std_fit, color='grey', alpha=0.2)
plt.fill_between(df['year'][sep_idx:], mu_test - 2*std_test, mu_test + 2*std_test, color='grey', alpha=0.2)
plt.legend(fontsize='small')
plt.title('Type II ML' + '\n' + 'RMSE: ' + str(rmse_) + '\n' + 'LPD: ' + str(lpd_), fontsize='small')

sig_var_1 =  gpr.kernel_.k2.k1.k1.constant_value
sig_var_3 = gpr.kernel_.k1.k1.k2.k1.k1.constant_value
sig_var_6 = gpr.kernel_.k1.k2.k1.constant_value
sig_var_9 = gpr.kernel_.k1.k1.k1.k1.constant_value
  
ls_2 = gpr.kernel_.k2.k1.k2.length_scale
ls_4 = gpr.kernel_.k1.k1.k2.k1.k2.length_scale
ls_5 =  gpr.kernel_.k1.k1.k2.k2.length_scale
ls_7 = gpr.kernel_.k1.k2.k2.length_scale
ls_10 = gpr.kernel_.k1.k1.k1.k2.length_scale
  
alpha_8 = gpr.kernel_.k1.k2.k2.alpha
noise_11 = gpr.kernel_.k2.k2.noise_level

ml_deltas = {'sig_var_1': sig_var_1, 'ls_2': ls_2, 'sig_var_3' : sig_var_3, 'ls_4': ls_4 , 'ls_5': ls_5 , 'sig_var_6': sig_var_6, 'ls_7': ls_7, 'alpha_8' : alpha_8, 'sig_var_9' : sig_var_9, 'ls_10' : ls_10, 'noise_11': noise_11}

ml_df = pd.DataFrame(data=ml_deltas, index=['ml'])

ml_df.to_csv('/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/co2_ml.csv', sep=',')

ml_df = pd.read_csv('/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Co2/co2_ml.csv', index_col=None)
#-------------------------------------------------------------------------------

# The MAP + HMC model

#-------------------------------------------------------------------------------

with pm.Model() as co2_model:
      
    # yearly periodic component x long term trend
    sig_var_3 = pm.HalfCauchy("sig_var_3", beta=2, testval=ml_deltas['sig_var_3'])
    ls_4 = pm.Gamma("ls_4", alpha=10, beta=2, testval=ml_deltas['ls_4'])
    ls_5 = pm.Gamma("ls_5", alpha=4, beta=3, testval=ml_deltas['ls_5'])
    cov_seasonal = sig_var_3*pm.gp.cov.Periodic(1, period=1, ls=ls_5) \
                            * pm.gp.cov.ExpQuad(1, ls_4)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

    # small/medium term irregularities
    sig_var_6 = pm.HalfCauchy("sig_var_6", beta=3, testval=ml_deltas['sig_var_6'])
    ls_7 = pm.Gamma("ls_7", alpha=2, beta=0.75, testval=ml_deltas['ls_7'])
    alpha_8 = pm.Gamma("alpha_8", alpha=5, beta=2, testval=ml_deltas['alpha_8'])
    cov_medium = sig_var_6*pm.gp.cov.RatQuad(1,ls_7, alpha_8)
    gp_medium = pm.gp.Marginal(cov_func=cov_medium)

    # long term trend
    sig_var_1 = pm.HalfCauchy("sig_var_1", beta=4, testval=ml_deltas['sig_var_1'])
    ls_2 = pm.Gamma("ls_2", alpha=4, beta=0.1, testval=ml_deltas['ls_2'])
    cov_trend = sig_var_1*pm.gp.cov.ExpQuad(1, ls_2)
    gp_trend = pm.gp.Marginal(cov_func=cov_trend)

    # noise model
    sig_var_9 = pm.HalfNormal("sig_var_9", sd=0.5, testval=ml_deltas['sig_var_9'])
    ls_10 = pm.Gamma("ls_10", alpha=2, beta=4, testval=ml_deltas['ls_10'])
    noise_11  = pm.HalfNormal("noise_11",  sd=0.25, testval=ml_deltas['noise_11'])
    cov_noise = sig_var_9*pm.gp.cov.ExpQuad(1, ls_10) +\
                pm.gp.cov.WhiteNoise(noise_11)

    # The Gaussian process is a sum of these three components
    gp = gp_seasonal + gp_medium + gp_trend

    # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
    y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=cov_noise)

with co2_model:
    # this line calls an optimizer to find the MAP
    mp = pm.find_MAP(include_transformed=True, progressbar=True)
    trace_hmc = pm.sample(tune=250, draws=300, chains=1, use_mmap=True)
    
with co2_model:
    
    pm.save_trace(trace_hmc, directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
    
with co2_model:
      
    trace_hmc_load = pm.load_trace(directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
        
      
    
names = [name for name in mp.keys() if not name.endswith('_')] 
mp = {}
for i in np.arange(len(names)):
      ml_deltas.update({names[i] : mp[names[i]]})

# Examining the fit of each of the additive GP components¶
#
## predict at a 15 day granularity
#dates = pd.date_range(start='3/15/1958', end="12/15/2003", freq="30D")
#tnew = dates_to_idx(dates)[:,None]
#
#print("Predicting with gp ...")
#mu, var = gp.predict(tnew, point=mp, diag=True)
#mean_pred = mu*std_co2 + first_co2
#var_pred  = var*std_co2**2
#
## make dataframe to store fit results
#fit = pd.DataFrame({"t": tnew.flatten(),
#                    "mu_total": mean_pred,
#                    "sd_total": np.sqrt(var_pred)},
#                   index=dates)
#
#print("Predicting with gp_trend ...")
#mu, var = gp_trend.predict(tnew, point=mp,
#                           given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
#                           diag=True)
#fit = fit.assign(mu_trend = mu*std_co2 + first_co2,
#                 sd_trend = np.sqrt(var*std_co2**2))
#
#print("Predicting with gp_medium ...")
#mu, var = gp_medium.predict(tnew, point=mp,
#                            given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
#                            diag=True)
#fit = fit.assign(mu_medium = mu*std_co2 + first_co2,
#                 sd_medium = np.sqrt(var*std_co2**2))
#
#print("Predicting with gp_seasonal ...")
#mu, var = gp_seasonal.predict(tnew, point=mp,
#                              given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
#                              diag=True)
#fit = fit.assign(mu_seasonal = mu*std_co2 + first_co2,
#                 sd_seasonal = np.sqrt(var*std_co2**2))
#print("Done")
#
### plot the components
#p = figure(title="Decomposition of the Mauna Loa Data",
#           x_axis_type='datetime', plot_width=550, plot_height=350)
#p.yaxis.axis_label = 'CO2 [ppm]'
#p.xaxis.axis_label = 'Date'
#
## plot mean and 2σ region of total prediction
#upper = fit.mu_total + 2*fit.sd_total
#lower = fit.mu_total - 2*fit.sd_total
#band_x = np.append(fit.index.values, fit.index.values[::-1])
#band_y = np.append(lower, upper[::-1])
#
## total fit
#p.line(fit.index, fit.mu_total,
#       line_width=1, line_color="firebrick", legend="Total fit")
#p.patch(band_x, band_y,
#        color="firebrick", alpha=0.6, line_color="white")
#
## trend
#p.line(fit.index, fit.mu_trend,
#       line_width=1, line_color="blue", legend="Long term trend")
#
## medium
#p.line(fit.index, fit.mu_medium,
#       line_width=1, line_color="green", legend="Medium range variation")
#
## seasonal
#p.line(fit.index, fit.mu_seasonal,
#       line_width=1, line_color="orange", legend="Seasonal process")
#
## true value
#p.circle(data_early.index, data_early['CO2'],
#         color="black", legend="Observed data")
#p.legend.location = "top_left"
#show(p)
#
## Prediction
#
#dates = pd.date_range(start="11/15/2003", end="6/15/2017", freq="1M")
#tnew = dates_to_idx(dates)[:,None]

print("Sampling gp predictions ...")
mu_pred, cov_pred = gp.predict(tnew, point=mp)

### plot mean and 2σ region of total prediction
# scale mean and var
mu_pred_sc = mu_pred * std_co2 + first_co2
sd_pred_sc = np.sqrt(np.diag(cov_pred) * std_co2**2 )


import matplotlib.pylab as plt

plt.figure()
plt.plot(fit.index, fit.mu_total, 'b', alpha=0.5, label='y_pred')
plt.fill_between(fit.index, fit.mu_total - 2*fit.sd_total, fit.mu_total + 2*fit.sd_total, color='grey', alpha=0.3)
plt.plot(dates, mu_pred_sc, 'b', alpha=0.5)
plt.fill_between(dates, mu_pred_sc - 2*sd_pred_sc, mu_pred_sc + 2*sd_pred_sc, color='grey', alpha=0.3)
plt.plot(data_early.index, data_early.CO2, 'bo', markersize=1, label='Train obs')
plt.plot(data_later.index, data_later.CO2, 'ro', markersize=1, label='Test obs')
plt.vlines(x=dates[0], ymin=325, ymax=420)
plt.legend()
plt.title('Type II ML')

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

             ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
             ls_4 = pm.Deterministic('ls_4', tt.exp(log_l4))
             ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
             ls_7 = pm.Deterministic('ls_7', tt.exp(log_l7))
             ls_10 = pm.Deterministic('ls_10', tt.exp(log_l10))
             
             # prior on amplitudes
             
             log_sv1 = pm.Uniform('log_sv1', lower=-10, upper=10)
             log_sv3 = pm.Uniform('log_sv3', lower=-10, upper=10)
             log_sv6 = pm.Uniform('log_sv6', lower=-10, upper=10)
             log_sv9 = pm.Uniform('log_sv9', lower=-10, upper=10)

             sig_var_1 = pm.Deterministic('sig_var_1', tt.exp(log_sv1))
             sig_var_3 = pm.Deterministic('sig_var_3', tt.exp(log_sv3))
             sig_var_6 = pm.Deterministic('sig_var_6', tt.exp(log_sv6))
             sig_var_9 = pm.Deterministic('sig_var_9', tt.exp(log_sv9))

             # prior on alpha
            
             log_alpha8 = pm.Uniform('log_alpha8', lower=-10, upper=10)
             alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
             
             # prior on noise variance term
            
             log_nv11 = pm.Uniform('log_nv11', lower=-5, upper=5)
             noise_11 = pm.Deterministic('noise_11', tt.exp(log_nv11))
               
             
             # Specify the covariance function
             
             k1 = pm.gp.cov.Constant(sig_var_1)*pm.gp.cov.ExpQuad(1, ls_2) 
             k2 = pm.gp.cov.Constant(sig_var_3)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
             k3 = pm.gp.cov.Constant(sig_var_6)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
             k4 = pm.gp.cov.Constant(sig_var_9)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(noise_11)
      
             k = k1 + k2 + k3 + k4
                
             gp = pm.gp.Marginal(cov_func=k)
                  
             # Marginal Likelihood
             y_ = gp.marginal_likelihood("y", X=t_train, y=y_train, noise=np.sqrt(noise_11))
             
             # HMC Nuts auto-tuning implementation
             trace_hmc = pm.sample(chains=1, start=ml_deltas)
             
      with hyp_learning:
            
              # this line calls an optimizer to find the MAP
              #mp = pm.find_MAP(include_transformed=True)
               fnew = gp.conditional("fnew", Xnew=t_test)
                  
      
      with hyp_learning:
            
             advi = pm.FullRankADVI()
             advi.fit()    
             trace_advi = advi.approx.sample(include_transformed=False) 


varnames = ['ls_2','ls_4','ls_5','ls_7','ls_10','sig_var_1','sig_var_3','sig_var_6','sig_var_9','alpha_8','noise_11']  
#varnames = [σ, ℓ_noise, η_noise, ℓ_trend, η_trend, α, ℓ_med, η_med, ℓ_psmooth , ℓ_pdecay, η_per]    

#varnames = [str(x) for x in varnames]   

def get_trace_df(trace_hmc, varnames):
      
    trace_df = pd.DataFrame()
    for i in varnames:
          trace_df[i] = trace_hmc.get_values(i)
    return trace_hmc

trace_df = get_trace_df(trace_hmc, varnames)

traces_part1 = pm.traceplot(trace_hmc, varnames[0:5], lines=ml_deltas)
traces_part2 = pm.traceplot(trace_hmc, varnames[5:], lines=ml_deltas)

for i in np.arange(5):
      
      delta = deltas.get(str(varnames[i]))
      xmin=0
      xmax = max(max(trace_hmc[varnames[i]]), delta)
      traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
      traces_part1[i][0].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
      traces_part1[i][0].axes.set_xlim(xmin, xmax)
      traces_part1[i][0].legend(fontsize='x-small')

for i in np.arange(6):
      
      delta = deltas.get(str(varnames[i+5]))
      traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
      traces_part2[i][0].hist(trace_hmc[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
      traces_part2[i][0].axes.set_xscale('log')
      traces_part2[i][0].legend(fontsize='x-small')
      
#dates = data

def get_posterior_predictive_gp_trace(trace, thin_factor, X_star):
      
      l = len(X_star)
      means_arr = np.empty(shape=(l,))
      std_arr = np.empty(shape=(l,))
      sq_arr = np.empty(shape=(l,))

      for i in np.arange(len(trace))[::thin_factor]:
            
            print(i)
            mu, var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            std = np.sqrt(np.diag(var))
            means_arr = np.vstack((mu, means_arr))
            std_arr = np.vstack((std, std_arr))
            sq_arr = np.vstack((np.multiply(mu, mu), sq_arr))
            
      final_mean = np.mean(means_arr[:-1,:], axis=0)
      final_std = np.mean(std_arr[:-1,:], axis=0) + np.mean(sq_arr[:-1,:], axis=0) - np.multiply(final_mean, final_mean)
      
      return final_mean, final_std, means_arr[:-1], std_arr[:-1], sq_arr[:-1]


pp_mean_hmc, pp_std_hm, means_arr, std_arr, sq_arr  = get_posterior_predictive_gp_trace(trace_hmc, 30, t_test) 

sample_means = (means_arr.T)*std_co2 + first_co2
mu_pred_hmc = pp_mean_hmc*std_co2 + first_co2
sd_pred_hmc = pp_std_hm*std_co2

# Plot with HMC w. ML II results

plt.figure()
plt.plot(dates, mu_pred_hmc)
plt.fill_between(dates, mu_pred_hmc - 2*sd_pred_hmc, mu_pred_hmc + 2*sd_pred_hmc, color='b', alpha=0.3)
plt.plot(fit.index, fit.mu_total, 'b', alpha=0.5)
plt.fill_between(fit.index, fit.mu_total - 2*fit.sd_total, fit.mu_total + 2*fit.sd_total, color='r', alpha=0.3)
plt.plot(dates, mu_pred_sc, 'b', alpha=0.5)
plt.fill_between(dates, mu_pred_sc - 2*sd_pred_sc, mu_pred_sc + 2*sd_pred_sc, color='grey', alpha=0.3)
plt.plot(data_early.index, data_early.CO2, 'ko', markersize=1, label='Train obs')
plt.plot(data_later.index, data_later.CO2, 'ko', markersize=1, label='Test obs')
plt.vlines(x=data_later.index[0], ymin=325, ymax=420)
plt.legend()


# Saving and reloading trace HMC

with model:      
      pm.save_trace(trace_hmc, directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/i_prior/')
      
with model:
      trace2 = pm.load_trace(directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/0')
      
# Write out trace summary & autocorrelation plots

prefix = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Config/'
summary_df = pm.summary(trace_hmc)
summary_df['Acc Rate'] = np.mean(trace_hmc.get_sampler_stats('mean_tree_accept'))
np.round(summary_df,3).to_csv(prefix + 'trace_summary_co2.csv')
      
# Plot with HMC results
      
      