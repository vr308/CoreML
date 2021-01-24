#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:47:45 2021

@author: vr308
"""

import numpy as np
import torch
from tqdm import trange
import torch.distributions as D
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from torch.autograd import Variable

torch.manual_seed(1)
np.random.seed(0)

class GaussMixture(torch.nn.Module):

    def __init__(self, K, dim):
        super().__init__()

        self.K = K
        self.dim = dim
        #self.raw_weights = torch.nn.Parameter(torch.randn(self.K))
        self.means = torch.nn.Parameter(torch.randn(self.K, self.dim))
        self.log_scales = torch.nn.Parameter(torch.zeros(self.K, self.dim))
        #self.scales = torch.exp(self.log_scales)
        
        self.concentration = torch.randint(high=100,size=(1,))
        self.alpha = torch.nn.Parameter(torch.tensor([self.concentration]*self.K).float())
        self.weights = D.Dirichlet(self.alpha)
        self.z = D.Categorical(self.weights)

    def log_prob(self, Y):
        return self.forward().log_prob(Y).sum()

    def forward(self, z):
         comp = D.Independent(D.Normal(self.means, self.log_scales.exp()), 1)
         return D.MixtureSameFamily(self.z, comp)

class BayesianGMM(torch.nn.Module):
    
    def __init__(self, Y, K, dim=2):
        '''
          :param Y (torch.tensor): data
          :param K (int): number of fixed components
          :param dim (int): the dimension of data space.
        
        '''
        super(BayesianGMM, self).__init__()
        
        self.Y = Y
        self.K = K
        self.dim = dim
                
        self.likelihood = GaussMixture(self.K, self.dim)
    
    def log_prior(self, Y):
        
        mu_prior = D.Normal(torch.zeros((self.K,self.dim)), torch.ones((self.K, self.dim))*3.0).rsample()
        scale_prior = D.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).rsample()
        self.means.log_prob(Y) + self.scales.log_prob(Y) + self.weights.log_prob(Y)
    
    def get_trainable_param_names(self):
      
      ''' Prints a list of parameters which will be 
      learnt in the process of optimising the objective '''
      
      for name, value in self.named_parameters():
          print(name)     
    
    def map_objective(self, Y):
        return self.likelihood.log_prob(Y) + self.log_prior()
      
    def train(self, Y, optimizer, n_steps):
    
        losses = np.zeros(n_steps)
        bar = trange(n_steps, leave=True)
        for step in bar:
            optimizer.zero_grad()
            loss = self.map_objective(Y)
            loss.backward()
            optimizer.step()
            losses[step] = loss
        return losses
    
    def plot_gmm(self, Y, labels):
        
        ells = [Ellipse(xy=self.means[i],width=3*self.scales[i][0], height=3*self.scales[i][1]) 
                    for i in range(self.K)]
        fig = plt.subplot(111, aspect='equal')
        plt.scatter(Y[:,0], Y[:,1], s=2, c=labels)
        for e in ells:
            print(e)
            fig.add_artist(e)
            e.set_clip_box(fig.bbox)
            e.set_alpha(0.5)
        plt.show()
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        
if __name__ == '__main__':
    
    # Generate some data
    
    Y, labels = skd.make_blobs(n_samples=2000, random_state=42,
                              cluster_std=[2.5, 0.5, 1.0])
    
    Y = torch.tensor(Y).float()
    
    model = BayesianGMM(Y, K=3)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = model.train(Y, optimizer, 4000)
    
    model.plot_gmm(Y, labels)