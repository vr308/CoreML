#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:24:56 2019

@author: vidhi
"""

import scipy.stats as st
import numpy as np
import pandas as pd

path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/'

# Load predicted means under different algorithms

# co2 

tag = 'Co2/pred_dist/'

mu_ml = pd.read_csv(path + tag + 'means_ml.csv')
samples_hmc = pd.read_csv(path + tag + 'means_hmc.csv')
samples_fr = pd.read_csv(path + tag + 'means_fr.csv')
samples_mf = pd.read_csv(path + tag + 'means_mf.csv')
samples_anlt = pd.read_csv(path + tag + 'means_anlt.csv')


mu_hmc = np.mean(samples_hmc) 
mu_mf = np.mean(samples_mf)
mu_fr = np.mean(samples_fr)


d = mu_mf - mu_fr
d_mean = np.mean(d)
d_std = np.std(d, axis=0, ddof=1)

d_mean/np.sqrt(se1**2 + se2**2)
d_mean/(d_std/np.sqrt(len(mu_hmc)))

def generate_p_value_matrix(mu_ml, mu_hmc, mu_mf, mu_fr, mu_anlt):
      
      return


def plot_p_matrix(ml_ml):

      return      



