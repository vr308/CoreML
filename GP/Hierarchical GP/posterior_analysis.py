#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:56:46 2019

@author: vidhi

Analysis for HMC / ADVI runs of hierarchical GP

"""

import csv
import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import seaborn as sns
import warnings
import advi_analysis as advi
warnings.filterwarnings("ignore")

# Helper functions for Trace analysis

def traceplots(trace, varnames, deltas, sep_idx):

      
            traces_part1 = pm.traceplot(trace, varnames[0:sep_idx], lines=deltas)
            traces_part2 = pm.traceplot(trace, varnames[sep_idx:], lines=deltas)
            
            for i in np.arange(sep_idx):
                  
                  delta = deltas.get(str(varnames[i]))
                  #xmax = max(max(trace[varnames[i]]), delta)
                  traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
                  traces_part1[i][0].hist(trace[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
                  traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
                  #traces_part1[i][0].axes.set_xlim(xmin, xmax)
                  traces_part1[i][0].legend(fontsize='x-small')
            
            for i in np.arange(sep_idx+1):
                  
                  delta = deltas.get(str(varnames[i+sep_idx]))
                  traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
                  traces_part2[i][0].hist(trace[varnames[i+sep_idx]], bins=100, normed=True, color='b', alpha=0.3)
                  traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
                  #traces_part2[i][0].axes.set_xscale('log')
                  traces_part2[i][0].legend(fontsize='x-small')


def traceplot_compare(mf, fr, trace_hmc, trace_mf, trace_fr, varnames, deltas, rv_mapping):

      traces_part1 = pm.traceplot(trace_hmc, varnames[0:5], lines=deltas)
      traces_part2 = pm.traceplot(trace_hmc, varnames[5:], lines=deltas)
      
      means_mf = mf.approx.bij.rmap(mf.approx.mean.eval())  
      std_mf = mf.approx.bij.rmap(mf.approx.std.eval())  
      
      means_fr = fr.approx.bij.rmap(fr.approx.mean.eval())  
      std_fr = fr.approx.bij.rmap(fr.approx.std.eval())  
      
      for i in np.arange(5):
            
            delta = deltas.get(str(varnames[i]))
            xmax = max(max(trace_hmc[varnames[i]]), delta)
            xmin = 0
            range_i = np.linspace(xmin, xmax, 1000)  
            mf_pdf = advi.get_implicit_variational_posterior(rv_mapping.get(varnames[i]), means_mf, std_mf, range_i)
            fr_pdf = advi.get_implicit_variational_posterior(rv_mapping.get(varnames[i]), means_fr, std_fr, range_i)
            traces_part1[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part1[i][0].hist(trace_hmc[varnames[i]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part1[i][1].axhline(y=delta, color='r', alpha=0.5)
            traces_part1[i][0].plot(range_i, mf_pdf, color='coral',alpha=0.4)
            traces_part1[i][0].fill_between(x=range_i, y1=[0]*len(range_i), y2=mf_pdf, color='coral', alpha=0.4)
            #traces_part1[i][0].hist(trace_fr[varnames[i]], bins=100, normed=True, color='green', alpha=0.3)
            traces_part1[i][0].plot(range_i, fr_pdf, color='g', alpha=0.4)
            traces_part1[i][0].fill_between(x=range_i, y1=[0]*len(range_i), y2=fr_pdf, color='green', alpha=0.4)
            #races_part1[i][0].axes.set_ylim(0, max(mf_pdf))
            traces_part1[i][0].legend(fontsize='x-small')
      
      for i in np.arange(6):
            
            delta = deltas.get(str(varnames[i+5]))
            xmax = max(max(trace_hmc[varnames[i+5]]), delta)
            xmin = 0
            range_i = np.linspace(xmin, xmax, 1000) 
            mf_pdf = advi.get_implicit_variational_posterior(rv_mapping.get(varnames[i+5]), means_mf, std_mf, range_i)
            fr_pdf = advi.get_implicit_variational_posterior(rv_mapping.get(varnames[i+5]), means_fr, std_fr, range_i)
            traces_part2[i][0].axvline(x=delta, color='r',alpha=0.5, label='ML ' + str(np.round(delta, 2)))
            traces_part2[i][0].hist(trace_hmc[varnames[i+5]], bins=100, normed=True, color='b', alpha=0.3)
            traces_part2[i][1].axhline(y=delta, color='r', alpha=0.5)
            traces_part2[i][0].plot(range_i, mf_pdf, color='coral',alpha=0.4)
            traces_part2[i][0].fill_between(x=range_i, y1=[0]*len(range_i), y2=mf_pdf, color='coral', alpha=0.4)
            traces_part2[i][0].plot(range_i, fr_pdf, color='g', alpha=0.4)
            traces_part2[i][0].fill_between(x=range_i, y1=[0]*len(range_i), y2=fr_pdf, color='green', alpha=0.4)
            #traces_part2[i][0].plot(ranges[i], get_implicit_variational_posterior(fr_rv[i], means_fr, std_fr, ranges[i]), color='g')
            traces_part2[i][0].legend(fontsize='x-small')
            

# Helper functions for bi-variate traces

      
def pair_grid_plot(trace_df, ml_deltas, varnames, color):

      g = sns.PairGrid(trace_df, vars=varnames, diag_sharey=False)
      g = g.map_lower(plot_bi_kde, ml_deltas=ml_deltas, color=color)
      g = g.map_diag(plot_hist, color=color)
      g = g.map_upper(plot_scatter, ml_deltas=ml_deltas, color=color)
      
      for i in np.arange(len(varnames)):
            g.axes[i,i].axvline(ml_deltas[g.x_vars[i]], color='r')
    
def plot_bi_kde(x,y, ml_deltas, color, label):
      
      sns.kdeplot(x, y, n_levels=20, color=color, shade=True, shade_lowest=False, bw='silverman')
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      
def plot_hist(x, color, label):
      
      sns.distplot(x, bins=100, color=color, kde=True)

def plot_scatter(x, y, ml_deltas, color, label):
      
      plt.scatter(x, y, c=color, s=0.5, alpha=0.7)
      plt.scatter(ml_deltas[x.name], ml_deltas[y.name], marker='x', color='r')
      
      
# Helper functions to generate posterior predictive samples 

def write_posterior_predictive_samples(trace, thin_factor, X, y, X_star, path, method):
      
      means_file = path + 'means_' + method + '.csv'
      std_file = path + 'std_' + method + '.csv'
      #trace_file = path + 'trace_' + method + '_' + str(len(X)) + '.csv'
          
      means_writer = csv.writer(open(means_file, 'w')) 
      std_writer = csv.writer(open(std_file, 'w'))
      #trace_writer = csv.writer(open(trace_file, 'w'))
      
      means_writer.writerow(X_star.flatten())
      std_writer.writerow(X_star.flatten())
      #trace_writer.writerow(varnames + ['lml'])
      
      for i in np.arange(len(trace))[::thin_factor]:
            
            print('Predicting ' + str(i))
            post_mean, post_var = gp.predict(X_star, point=trace[i], pred_noise=False, diag=True)
            post_std = np.sqrt(post_var)
            #K, K_s, K_ss, K_noise, K_inv = get_kernel_matrix_blocks(X, X_star, len(X), trace[i])
            #post_mean, post_std = analytical_gp(y, K, K_s, K_ss, K_noise, K_inv)
            #marginal_likelihood = compute_log_marginal_likelihood(K_noise, y)
            #mu, var = pm.gp.Marginal.predict(Xnew=X_star, point=trace[i], pred_noise=False, diag=True)
            #std = np.sqrt(var)
            #list_point = [trace[i]['sig_sd'], trace[i]['ls'], trace[i]['noise_sd'], marginal_likelihood]
            
            print('Writing out ' + str(i) + ' predictions')
            means_writer.writerow(np.round(post_mean, 3))
            std_writer.writerow(np.round(post_std, 3))
            #trace_writer.writerow(np.round(list_point, 3))


def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):
      
      # Fixed at 95% CI
      
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True)
            mixture_draws = np.array([st.norm.rvs(loc=sample_means.iloc[j,i], scale=sample_stds.iloc[j,i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5,97.5])
            lower_.append(lower)
            upper_.append(upper)
      return np.array(lower_), np.array(upper_)

def get_posterior_predictive_mean(sample_means):
      
    return np.mean(sample_means)

# Helper functions for posterior predictive checks


# TODO


# Helper functions for Metrics

      
def rmse(post_mean, y_test):
    
    return np.round(np.sqrt(np.mean(np.square(post_mean - y_test))),3)

def log_predictive_density(y_test, list_means, list_stds):
      
      lppd_per_point = []
      for i in np.arange(len(y_test)):
            print(i)
            lppd_per_point.append(st.norm.pdf(y_test[i], list_means[i], list_stds[i]))
      return np.round(np.mean(np.log(lppd_per_point)),3)
            
def log_predictive_mixture_density(y_test, list_means, list_std):
      
      lppd_per_point = []
      for i in np.arange(len(y_test)):
            print(i)
            components = []
            for j in np.arange(len(list_means)):
                  components.append(st.norm.pdf(y_test[i], list_means.iloc[:,i][j], list_std.iloc[:,i][j]))
            lppd_per_point.append(np.mean(components))
      return lppd_per_point, np.round(np.mean(np.log(lppd_per_point)),3)