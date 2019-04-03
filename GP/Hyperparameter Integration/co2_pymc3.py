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

from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Span, Label, Legend
from bokeh.io import output_notebook
from bokeh.palettes import brewer
output_notebook()

data_monthly = pd.read_csv(pm.get_data("monthly_in_situ_co2_mlo.csv"), header=56)

# - replace -99.99 with NaN
data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)

# fix column names
cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]
data_monthly.columns = cols
cols.remove("--"); cols.remove("--")
data_monthly = data_monthly[cols]

data_monthly.dropna(inplace=True)

# fix time index
data_monthly["day"] = 15
data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
cols.remove("year"); cols.remove("month")
data_monthly = data_monthly[cols]

data_monthly.head(5)

def dates_to_idx(timelist):
    reference_time = pd.to_datetime('1958-03-15')
    t = (timelist - reference_time) / pd.Timedelta(1, "Y")
    return np.asarray(t)

t = dates_to_idx(data_monthly.index)

# normalize CO2 levels
y = data_monthly["CO2"].values
first_co2 = y[0]
std_co2 = np.std(y)
y_n = (y - first_co2) / std_co2

data_monthly = data_monthly.assign(t = t)
data_monthly = data_monthly.assign(y_n = y_n)

sep_idx = data_monthly.index.searchsorted(pd.to_datetime("2003-12-15"))
data_early = data_monthly.iloc[:sep_idx+1, :]
data_later = data_monthly.iloc[sep_idx:, :]

p = figure(x_axis_type='datetime', title='Monthly CO2 Readings from Mauna Loa',
           plot_width=550, plot_height=350)
p.yaxis.axis_label = 'CO2 [ppm]'
p.xaxis.axis_label = 'Date'
predict_region = BoxAnnotation(left=pd.to_datetime("2003-12-15"),
                               fill_alpha=0.1, fill_color="firebrick")
p.add_layout(predict_region)
ppm400 = Span(location=400,
              dimension='width', line_color='red',
              line_dash='dashed', line_width=2)
p.add_layout(ppm400)

p.line(data_monthly.index, data_monthly['CO2'],
       line_width=2, line_color="black", alpha=0.5)
p.circle(data_monthly.index, data_monthly['CO2'],
         line_color="black", alpha=0.1, size=2)

train_label = Label(x=100, y=165, x_units='screen', y_units='screen',
                    text='Training Set', render_mode='css', border_line_alpha=0.0,
                    background_fill_alpha=0.0)
test_label  = Label(x=585, y=80, x_units='screen', y_units='screen',
                    text='Test Set', render_mode='css', border_line_alpha=0.0,
                    background_fill_alpha=0.0)

p.add_layout(train_label)
p.add_layout(test_label)
show(p)

# pull out normalized data
t = data_early["t"].values[:,None]
y = data_early["y_n"].values

# The model

with pm.Model() as model:
      
    # yearly periodic component x long term trend
    η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
    ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
    period  = pm.Normal("period", mu=1, sd=0.05)
    ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
    cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, period, ℓ_psmooth) \
                            * pm.gp.cov.Matern52(1, ℓ_pdecay)
    gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

    # small/medium term irregularities
    η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
    ℓ_med = pm.Gamma("ℓ_med", alpha=2, beta=0.75)
    α = pm.Gamma("α", alpha=5, beta=2)
    cov_medium = η_med**2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
    gp_medium = pm.gp.Marginal(cov_func=cov_medium)

    # long term trend
    η_trend = pm.HalfCauchy("η_trend", beta=2, testval=2.0)
    ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
    cov_trend = η_trend**2 * pm.gp.cov.ExpQuad(1, ℓ_trend)
    gp_trend = pm.gp.Marginal(cov_func=cov_trend)

    # noise model
    η_noise = pm.HalfNormal("η_noise", sd=0.5, testval=0.05)
    ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
    σ  = pm.HalfNormal("σ",  sd=0.25, testval=0.05)
    cov_noise = η_noise**2 * pm.gp.cov.Matern32(1, ℓ_noise) +\
                pm.gp.cov.WhiteNoise(σ)

    # The Gaussian process is a sum of these three components
    gp = gp_seasonal + gp_medium + gp_trend

    # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
    y_ = gp.marginal_likelihood("y", X=t, y=y, noise=cov_noise)

    # this line calls an optimizer to find the MAP
    mp = pm.find_MAP(include_transformed=True)
    
sorted([name+":"+str(mp[name]) for name in mp.keys() if not name.endswith("_")])

# Examining the fit of each of the additive GP components¶

# predict at a 15 day granularity
dates = pd.date_range(start='3/15/1958', end="12/15/2003", freq="30D")
tnew = dates_to_idx(dates)[:,None]

print("Predicting with gp ...")
mu, var = gp.predict(tnew, point=mp, diag=True)
mean_pred = mu*std_co2 + first_co2
var_pred  = var*std_co2**2

# make dataframe to store fit results
fit = pd.DataFrame({"t": tnew.flatten(),
                    "mu_total": mean_pred,
                    "sd_total": np.sqrt(var_pred)},
                   index=dates)

print("Predicting with gp_trend ...")
mu, var = gp_trend.predict(tnew, point=mp,
                           given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
                           diag=True)
fit = fit.assign(mu_trend = mu*std_co2 + first_co2,
                 sd_trend = np.sqrt(var*std_co2**2))

print("Predicting with gp_medium ...")
mu, var = gp_medium.predict(tnew, point=mp,
                            given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
                            diag=True)
fit = fit.assign(mu_medium = mu*std_co2 + first_co2,
                 sd_medium = np.sqrt(var*std_co2**2))

print("Predicting with gp_seasonal ...")
mu, var = gp_seasonal.predict(tnew, point=mp,
                              given={"gp": gp, "X": t, "y": y, "noise": cov_noise},
                              diag=True)
fit = fit.assign(mu_seasonal = mu*std_co2 + first_co2,
                 sd_seasonal = np.sqrt(var*std_co2**2))
print("Done")

## plot the components
p = figure(title="Decomposition of the Mauna Loa Data",
           x_axis_type='datetime', plot_width=550, plot_height=350)
p.yaxis.axis_label = 'CO2 [ppm]'
p.xaxis.axis_label = 'Date'

# plot mean and 2σ region of total prediction
upper = fit.mu_total + 2*fit.sd_total
lower = fit.mu_total - 2*fit.sd_total
band_x = np.append(fit.index.values, fit.index.values[::-1])
band_y = np.append(lower, upper[::-1])

# total fit
p.line(fit.index, fit.mu_total,
       line_width=1, line_color="firebrick", legend="Total fit")
p.patch(band_x, band_y,
        color="firebrick", alpha=0.6, line_color="white")

# trend
p.line(fit.index, fit.mu_trend,
       line_width=1, line_color="blue", legend="Long term trend")

# medium
p.line(fit.index, fit.mu_medium,
       line_width=1, line_color="green", legend="Medium range variation")

# seasonal
p.line(fit.index, fit.mu_seasonal,
       line_width=1, line_color="orange", legend="Seasonal process")

# true value
p.circle(data_early.index, data_early['CO2'],
         color="black", legend="Observed data")
p.legend.location = "top_left"
show(p)

# Prediction

dates = pd.date_range(start="11/15/2003", end="12/15/2020", freq="10D")
tnew = dates_to_idx(dates)[:,None]

print("Sampling gp predictions ...")
mu_pred, cov_pred = gp.predict(tnew, point=mp)

### plot mean and 2σ region of total prediction
# scale mean and var
mu_pred_sc = mu_pred * std_co2 + first_co2
sd_pred_sc = np.sqrt(np.diag(cov_pred) * std_co2**2 )


import matplotlib.pylab as plt

plt.figure()
#plt.plot(fit.index, fit.mu_total)
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
             y_ = gp.marginal_likelihood("y", X=t, y=y, noise=np.sqrt(noise_var))
             
             # HMC Nuts auto-tuning implementation
             
             trace_hmc = pm.sample(chains=1)
             
      with hyp_learning:
            
              # this line calls an optimizer to find the MAP
              mp = pm.find_MAP(include_transformed=True)
             
      
      with hyp_learning:
            
             advi = pm.FullRankADVI()
             advi.fit()    
             trace_advi = advi.approx.sample(include_transformed=False) 


varnames = ['l2','l4','l5','l7','l10','sig_var1','sig_var3','sig_var6','sig_var9','alpha8','noise_var']         

def get_trace_df(trace_hmc, varnames):
      
    trace_df = pd.DataFrame()
    for i in varnames:
          trace_df[i] = trace_hmc.get_values(i)
    return trace_hmc

trace_df = get_trace_df(trace_hmc, varnames)
deltas = mp_sub = {key : mp.get(key) for key in varnames}

traces_part1 = pm.traceplot(trace_hmc, varnames[0:5], lines=deltas)
traces_part2 = pm.traceplot(trace_hmc, varnames[5:], lines=deltas)

for i in np.arange(5):
      
      xmin=-0.02
      xmax = max(max(trace_hmc[varnames[i]]), deltas.get(varnames[i])) + 5 
      traces_part1[i][0].axvline(x=deltas.get(varnames[i]), color='r',alpha=0.5, label='ML ' + str(np.round(deltas.get(varnames[i]), 2)))
      traces_part1[i][0].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part1[i][1].axhline(y=deltas.get(varnames[i]), color='r', alpha=0.5)
      traces_part1[i][0].axes.set_xlim(xmin, xmax)
      traces_part1[i][0].legend(fontsize='x-small')

for i in np.arange(6):
      
      traces_part2[i][0].axvline(x=deltas.get(varnames[i+5]), color='r',alpha=0.5, label='ML ' + str(np.round(deltas.get(varnames[i+5]), 2)))
      traces_part2[i][0].hist(trace_hmc[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
      traces_part2[i][1].axhline(y=deltas.get(varnames[i+5]), color='r', alpha=0.5)
      traces_part2[i][0].axes.set_xscale('log')
      traces_part2[i][0].legend(fontsize='x-small')
      

# predict at a 15 day granularity
dates = pd.date_range(start='3/15/1958', end="12/15/2019", freq="30D")
tnew = dates_to_idx(dates)[:,None]

def get_posterior_predictive_gp_trace(trace, thin_factor, X_star):
      
      l = len(X_star)
      means_arr = np.empty(shape=(l,))
      std_arr = np.empty(shape=(l,))
      #sq_arr = np.empty(shape=(l,))

      for i in np.arange(len(trace))[::thin_factor]:
            print(i)
            mu, var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            std = np.sqrt(var)
            means_arr = np.vstack((mu, means_arr))
            std_arr = np.vstack((std, std_arr))
            #sq_arr = np.vstack((sq_arr, np.multiply(mu, mu)))
            
      final_mean = np.mean(means_arr[:-1,:], axis=0)O
      final_std = np.mean(std_arr[:-1,:], axis=0) #+ np.mean(sq_arr[:-1,:], axis=0) - np.multiply(final_mean, final_mean)
      return final_mean, final_std
      
pp_mean_hmc, pp_std_hmc  = get_posterior_predictive_gp_trace(trace_hmc, 10, tnew) 

mu_pred_hmc = pp_mean_hmc*std_co2 + first_co2
sd_pred_hmc = pp_std_hmc*std_co2


plt.figure()
plt.plot(dates, mu_pred_hmc)
plt.fill_between(dates, mu_pred_hmc - 2*sd_pred_hmc, mu_pred_hmc + 2*sd_pred_hmc, color='grey', alpha=0.3)

plt.plot(fit.index, fit.mu_total, 'b', alpha=0.5, label='y_pred')
plt.fill_between(fit.index, fit.mu_total - 2*fit.sd_total, fit.mu_total + 2*fit.sd_total, color='grey', alpha=0.3)
plt.plot(dates, mu_pred_sc, 'b', alpha=0.5)
plt.fill_between(dates, mu_pred_sc - 2*sd_pred_sc, mu_pred_sc + 2*sd_pred_sc, color='grey', alpha=0.3)
plt.plot(data_early.index, data_early.CO2, 'bo', markersize=1, label='Train obs')
plt.plot(data_later.index, data_later.CO2, 'ro', markersize=1, label='Test obs')
plt.vlines(x=dates[0], ymin=325, ymax=420)
plt.legend()

with hyp_learning:
      
      pm.save_trace(trace_hmc, directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/')
      
with hyp_learning:
      trace2 = pm.load_trace(directory='/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Traces_pickle/')
