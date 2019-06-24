#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:57:52 2019

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
import seaborn as sns
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
warnings.filterwarnings("ignore")
import csv

if __name__ == "__main__":


      varnames = ['s_1', 'ls_2', 's_3', 'ls_4', 'ls_5', 's_6', 'ls_7', 'alpha_8', 's_9', 'ls_10', 'n_11']
      

      home_path = '~/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Airline/'
      uni_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Data/Airline/'
      results_path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hyperparameter Integration/Results/Airline/'
      
      path = uni_path
      
      df = pd.read_csv(path + 'AirPassengers.csv', names=['Month', 'Passengers'], infer_datetime_format=True,  sep=',', keep_default_na=False, header=0, parse_dates=True)
      
      mean = df['Passengers'][0]
      std = np.std(df['Passengers'])   
      
      lambda x: 
      
      # normalize co2 levels
         
      #y = normalize(df['co2'])
      y = df['Passengers']
      t = df['Month'] - df['Month'][0]
      
      sep_idx = 95
      
      y_train = y[0:sep_idx].values
      y_test = y[sep_idx:].values
      
      t_train = t[0:sep_idx].values[:,None]
      t_test = t[sep_idx:].values[:,None]
      