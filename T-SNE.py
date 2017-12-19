#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:46:10 2017

@author: vr308
"""

from sklearn.manifold import TSNE
import sklearn.datasets as skd

data, targets = skd.load_wine(return_X_y = True)
X_embedded = TSNE(n_components=2).fit_transform(data)
