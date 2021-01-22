#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:33:53 2021

@author: vr308

Fitting a parametric GMM with fixed components / Bayesian GMM where the number of components is learnt.

# Log losses, and changing parameters

"""

import numpy as np
import torch
from tqdm import trange
import torch.distributions as D
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import sklearn.datasets as skd

class GMM(torch.nn.Module):
    
    def __init__(self, Y, K, dim=2):
        '''
          :param K (int): number of fixed components
          :param D (int): the dimension of data space.
        
        '''
        super(GMM, self).__init__()
        
        self.Y = Y
        self.K = K
        self.dim = dim
        
        ind = np.random.choice(len(Y), size=self.K)
        
        self.raw_weights = torch.nn.Parameter(torch.randn(self.K))
        self.means = torch.nn.Parameter(Y[ind])
        self.log_scales = torch.nn.Parameter(torch.randn(self.K, self.dim))
        
        self._update()
        
    def _update(self):
           
        self.weights = torch.nn.functional.softmax(self.raw_weights)
        self.scales = torch.exp(self.log_scales)
        self.z = D.Categorical(self.weights)

    
    def forward(self, Y):
        
        #comp = D.MultivariateNormal(model.means, torch.reshape(model.scales,(self.K,self.dim,1))*torch.eye(self.dim))
        comp = D.Independent(D.Normal(model.means, model.scales), 1)
        return D.MixtureSameFamily(self.z, comp).log_prob(Y)
        
    def initialise(self):
        return
    
    def get_trainable_param_names(self):
      
      ''' Prints a list of parameters which will be 
      learnt in the process of optimising the objective '''
      
      for name, value in self.named_parameters():
          print(name)     
    
    def negative_log_loss(self,Y):
        return -self.forward(Y).mean()
      
    def train(self, Y, optimizer, n_steps):
    
        losses = np.zeros(n_steps)
        bar = trange(n_steps, leave=True)
        for step in bar:
            optimizer.zero_grad()
            self._update()
            loss = self.negative_log_loss(Y)
            loss.backward(retain_graph=True)
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
    
    model = GMM(Y, K=3)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = model.train(Y, optimizer, 4000)
    
    model.plot_gmm(Y, labels)