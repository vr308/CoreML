#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:16:36 2016

@author: vidhi
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# Datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine


# Data preprocessing and machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# To measure performance
from sklearn import metrics

###############################################################################
#                2. BDTs vs BXTs                
###############################################################################

# Load data and store it into pandas DataFrame objects
iris = load_iris()
breast_cancer = load_breast_cancer()
wine = load_wine()

data = wine

X = pd.DataFrame(data.data[:, :], columns = data.feature_names[:])
y = pd.DataFrame(data.target, columns =["Species"])

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50, random_state = 100)

param_grid_dt = {
     'learning_rate': [0.2, 0.4, 1],
     'base_estimator__max_depth': [10, 20, 50, 100],
    'base_estimator__max_features': [2, 3, 5, 10],
    'base_estimator__min_samples_leaf': [5, 10, 20],
    'base_estimator__min_samples_split': [10, 20 , 30],
    'n_estimators': [5,10,20],
}

param_grid_xt = {
    'learning_rate': [0.2, 0.4],
    'n_estimators': [5,10,20],
    'base_estimator__max_samples': [50, 100],
    'base_estimator__n_estimators': [20, 50],
    'base_estimator__base_estimator__max_features': [2, 3, 5],
    'base_estimator__base_estimator__min_samples_leaf': [10, 20],
    'base_estimator__base_estimator__min_samples_split': [10, 20],
    'base_estimator__base_estimator__max_depth': [5, 10, 20], 
}


# Defining the base classifiers 

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=7, max_features='auto')
x_trees = BaggingClassifier(base_estimator=ExtraTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=7, max_features='auto'), n_estimators=20, max_samples=100)


bdt = AdaBoostClassifier(base_estimator=tree, n_estimators=10)
bdt_cv = GridSearchCV(estimator=bdt, param_grid = param_grid_dt, refit=True, cv=5)
bdt_cv.fit(X_train, np.ravel(y_train))

bxt = AdaBoostClassifier(base_estimator=x_trees, n_estimators=10)
bxt_cv = GridSearchCV(estimator=bxt, param_grid=param_grid_xt, refit=True, cv=2)
bxt_cv.fit(X_train, np.ravel(y_train))

print(bdt_cv.best_score_)
print(bxt_cv.best_score_)

# Creating an ensemble 
bdt = []
bxt = []

stages = [5, 7, 10, 15, 20, 30, 40, 50]

for n in stages:
      print(n)
      bdt.append(AdaBoostClassifier(base_estimator=tree, n_estimators=n))
      bxt.append(AdaBoostClassifier(base_estimator=x_trees, n_estimators=n))

# Training classifiers

bdt_trained = []
bxt_trained = []

bdt_acc = []
bxt_acc = []

for n in np.arange(len(stages)):
      
      bdt_trained.append(bdt[n].fit(X_train, np.ravel(y_train)))
      bxt_trained.append(bxt[n].fit(X_train, np.ravel(y_train)))
      
      bdt_pred = bdt_trained[n].predict(X_test)
      bxt_pred = bxt_trained[n].predict(X_test)
      
      bdt_acc.append(metrics.accuracy_score(y_test, bdt_pred))
      bxt_acc.append(metrics.accuracy_score(y_test, bxt_pred))
      
  

plt.figure()
plt.plot(stages, bdt_acc)
plt.plot(stages, bxt_acc)


