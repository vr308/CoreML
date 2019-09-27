#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:24:56 2019

@author: vidhi
"""

import scipy.stats as st
from itertools import combinations
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

path = '/home/vidhi/Desktop/Workspace/CoreML/GP/Hierarchical GP/Results/'

# Load predicted means under different algorithms

datasets = ['CO2', 'Airline', 'Power','Wine', 'Concrete', 'Energy','Physics']

def generate_p_value_matrix(path, tag):

      mu_ml = pd.read_csv(path + tag + '/pred_dist/means_ml.csv', header=None)
      mu_hmc = pd.read_csv(path + tag + '/pred_dist/means_hmc.csv', header=None)
      mu_fr = pd.read_csv(path + tag + '/pred_dist/means_fr.csv', header=None)
      mu_mf = pd.read_csv(path + tag + '/pred_dist/means_mf.csv', header=None)
      #samples_anlt = pd.read_csv(path + tag + 'means_anlt.csv')
      
      
      # Create a dictionary 
      
      means = dict({'ML-II': np.array(mu_ml).ravel(), 
                   'HMC': np.array(mu_hmc).ravel(),
                   'MF-VI': np.array(mu_mf).ravel(),
                   'FR-VI': np.array(mu_fr).ravel()})
      
      
      index = ['ML-II','HMC', 'MF-VI', 'FR-VI']
      
      p_matrix = pd.DataFrame(index=index, columns=index, dtype=np.float32)
            
      for i in index:
            p_matrix[i][i] = np.NAN
      
      for i in combinations(index, 2):
            p_matrix[i[0]][i[1]] = st.ttest_rel(means[i[0]], means[i[1]])[1]
            p_matrix[i[1]][i[0]] = st.ttest_rel(means[i[0]], means[i[1]])[1]
      
      sns.heatmap(p_matrix, annot=True)
      plt.title(tag, fontsize='small')
      
      
generate_p_value_matrix(path, 'Physics')
      
for i in datasets:
      
      tag = 
  
plt.figure(figsize=(18,3))
plt.subplot(141)
generate_p_value_matrix(p_matrix, means, title=i)
plt.subplot(142)
generate_p_value_matrix(p_matrix, means, title=i)

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x, y), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


