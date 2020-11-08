#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL Colab notebook for long form paper
"""

import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import matplotlib
import mlai
from sklearn import svm
import os
import pods
from ipywidgets import IntSlider
import daft
from matplotlib import rc

rc("font", **{'family':'sans-serif','sans-serif':['Helvetica']}, size=30)
#rc("text", usetex=True)

font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


def create_data(per_cluster=30):
    """Create a randomly sampled data set
    
    :param per_cluster: number of points in each cluster
    """
    X = []
    y = []
    scale = 3
    prec = 1/(scale*scale)
    pos_mean = [[-1, 0],[0,0.5],[1,0]]
    pos_cov = [[prec, 0.], [0., prec]]
    neg_mean = [[0, -0.5],[0,-0.5],[0,-0.5]]
    neg_cov = [[prec, 0.], [0., prec]]
    for mean in pos_mean:
        X.append(np.random.multivariate_normal(mean=mean, cov=pos_cov, size=per_class))
        y.append(np.ones((per_class, 1)))
    for mean in neg_mean:
        X.append(np.random.multivariate_normal(mean=mean, cov=neg_cov, size=per_class))
        y.append(np.zeros((per_class, 1)))
    return np.vstack(X), np.vstack(y).flatten()

#############


def plot_contours(ax, cl, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    :param ax: matplotlib axes object
    :param cl: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param params: dictionary of params to pass to contourf, optional
    """
    Z = cl.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot decision boundary and regions
    out = ax.contour(xx, yy, Z, 
                     levels=[-1., 0., 1], 
                     colors='black', 
                     linestyles=['dashed', 'solid', 'dashed'])
    out = ax.contourf(xx, yy, Z, 
                     levels=[Z.min(), 0, Z.max()], 
                     colors=[[0.5, 1.0, 0.5], [1.0, 0.5, 0.5]])
    return out

#############

def decision_boundary_plot(models, X, y, axs, filename, directory, titles, xlim, ylim):
    """Plot a decision boundary on the given axes
    
    :param axs: the axes to plot on.
    :param models: the SVM models to plot
    :param titles: the titles for each axis
    :param X: input training data
    :param y: target training data"""
    for ax in axs.flatten():
        ax.clear()
    X0, X1 = X[:, 0], X[:, 1]
    if xlim is None:
        xlim = [X0.min()-1, X0.max()+1]
    if ylim is None:
        ylim = [X1.min()-1, X1.max()+1]
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], 0.02),
                         np.arange(ylim[0], ylim[1], 0.02))
    for cl, title, ax in zip(models, titles, axs.flatten()):
        plot_contours(ax, cl, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.plot(X0[y==1], X1[y==1], 'r.', markersize=10)
        ax.plot(X0[y==0], X1[y==0], 'g.', markersize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        mlai.write_figure(filename=filename,
                          directory=directory,
                          figure=fig,
                          transparent=True)
    return xlim, ylim

# Create an instance of SVM and fit the data. 
C = 100.0  # SVM regularization parameter
gammas = [0.001, 0.01, 0.1, 1]


per_class=30
num_samps = 20
# Set-up 2x2 grid for plotting.
fig, ax = plt.subplots(1, 4, figsize=(10,3))
xlim=None
ylim=None
for samp in range(num_samps):
    X, y=create_data(per_class)
    models = []
    titles = []
    for gamma in gammas:
        models.append(svm.SVC(kernel='rbf', gamma=gamma, C=C))
        titles.append('$\gamma={}$'.format(gamma))
    models = (cl.fit(X, y) for cl in models)
    xlim, ylim = decision_boundary_plot(models, X, y, 
                           axs=ax, 
                           filename='bias-variance{samp:0>3}.svg'.format(samp=samp), 
                           directory= 'ml',
                           titles=titles,
                          xlim=xlim,
                          ylim=ylim)
    
#############


pods.notebook.display_plots('bias-variance{samp:0>3}.svg', 
                            directory='./ml', 
                            samp=IntSlider(0,0,10,1))

#################

pgm = daft.PGM(shape=[2, 1],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)

pgm.add_node(daft.Node("y", r"$\mathbf{y}$", 0.5, 0.5, fixed=False, observed=True))

pgm.render().figure.savefig("./ml/y-only-graph.svg", transparent=True)

#################


pgm = daft.PGM(shape=[2, 2],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)

pgm.add_node(daft.Node("y", r"$\mathbf{y}$", 0.5, 0.5, fixed=False, observed=True))
pgm.add_node(daft.Node("u", r"$\mathbf{u}$", 0.5, 1.5, fixed=False))
pgm.add_node(daft.Node("ystar", r"$\mathbf{y}^*$", 1.5, 0.5, fixed=False))
pgm.add_node(daft.Node("ustar", r"$\mathbf{u}^*$", 1.5, 1.5, fixed=False))

pgm.add_edge("u", "y", directed=False)
pgm.add_edge("ustar", "y", directed=False)
pgm.add_edge("u", "ustar", directed=False)
pgm.add_edge("ystar", "y", directed=False)
pgm.add_edge("ustar", "ystar", directed=False)
pgm.add_edge("u", "ystar", directed=False)

pgm.render().figure.savefig("./ml/u-to-y-ustar-to-y.svg", transparent=True)

#################

pgm = daft.PGM(shape=[2, 3],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)

pgm.add_node(daft.Node("y", r"$\mathbf{y}$", 0.5, 0.5, fixed=False, observed=True))
pgm.add_node(daft.Node("f", r"$\mathbf{f}$", 0.5, 1.5, fixed=False))
pgm.add_node(daft.Node("u", r"$\mathbf{u}$", 0.5, 2.5, fixed=False))
pgm.add_node(daft.Node("ustar", r"$\mathbf{u}^*$", 1.5, 2.5, fixed=False))

pgm.add_edge("u", "f", directed=False)
pgm.add_edge("f", "y")
pgm.add_edge("ustar", "f", directed=False)
pgm.add_edge("u", "ustar", directed=False)

pgm.render().figure.savefig("./ml/u-to-f-to-y-ustar-to-f.svg", transparent=True)

###################


pgm = daft.PGM(shape=[2, 3],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)
reduce_alpha={"alpha": 0.3}
pgm.add_node(daft.Node("y", r"$y_i$", 0.5, 0.5, fixed=False, observed=True))
pgm.add_node(daft.Node("f", r"$f_i$", 0.5, 1.5, fixed=False))
pgm.add_node(daft.Node("u", r"$\mathbf{u}$", 0.5, 2.5, fixed=False))
pgm.add_node(daft.Node("ustar", r"$\mathbf{u}^*$", 1.5, 1.5, fixed=False, plot_params=reduce_alpha))
pgm.add_plate([0.125, 0.125, 0.75, 1.75], label=r"$i=1\dots n$", fontsize=18)

pgm.add_edge("f", "y")
pgm.add_edge("u", "f")
pgm.add_edge("ustar", "f", plot_params=reduce_alpha)

pgm.render().figure.savefig("./ml/u-to-f_i-to-y_i.svg", transparent=True)

#####

pgm = daft.PGM(shape=[2, 3],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)
reduce_alpha={"alpha": 0.3}
pgm.add_node(daft.Node("y", r"$y_i$", 0.5, 0.5, fixed=False, observed=True))
pgm.add_node(daft.Node("f", r"$f_i$", 0.5, 1.5, fixed=False))
pgm.add_node(daft.Node("u", r"$\mathbf{u}$", 0.5, 2.5, fixed=True))
pgm.add_node(daft.Node("ustar", r"$\mathbf{u}^*$", 1.5, 1.5, fixed=True, plot_params=reduce_alpha))
pgm.add_plate([0.125, 0.125, 0.75, 1.75], label=r"$i=1\dots n$", fontsize=18)

pgm.add_edge("f", "y")
pgm.add_edge("u", "f")
pgm.add_edge("ustar", "f", plot_params=reduce_alpha)

pgm.render().figure.savefig("./ml/given-u-to-f_i-to-y_i.svg", transparent=True)

pgm = daft.PGM(shape=[2, 3],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)
reduce_alpha={"alpha": 0.3}
pgm.add_node(daft.Node("y", r"$y_i$", 0.5, 0.5, fixed=False, observed=True))
pgm.add_node(daft.Node("f", r"$f_i$", 0.5, 1.5, fixed=False))
pgm.add_node(daft.Node("theta", r"$\boldsymbol{\theta}$", 0.5, 2.5, fixed=True))
#e, plot_params=reduce_alpha))
pgm.add_plate([0.125, 0.125, 0.75, 1.75], label=r"$i=1\dots n$", fontsize=18)

pgm.add_edge("f", "y")
pgm.add_edge("theta", "f")

pgm.render().figure.savefig("./ml/given-theta-to-f_i-to-y_i.svg", transparent=True)