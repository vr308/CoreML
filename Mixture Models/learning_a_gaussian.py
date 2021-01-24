#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:01:00 2021

@author: vr308

Fitting a Gaussian in pytorch
"""
import torch
import numpy as np
import torch.distributions as D
from tqdm import trange
from torch.distributions import Normal, kl_divergence

class LearningGaussianDist(torch.nn.Module):
    
    def __init__(self):
            super(LearningGaussianDist, self).__init__()
            
            # MAP Inference / Point estimate learning
            
            self.mu = torch.nn.Parameter(torch.tensor([0.0]))
            self.sigma = torch.nn.Parameter(torch.tensor([1.0]))
            #self.sigma = torch.exp(self.log_sigma)
            #self.update_params()
            self.mu_prior = D.Normal(0.0, 1.0)
            self.scale_prior = D.Gamma(1.0, 1.0)
    
    def forward(self):
        
        return D.Normal(self.mu, self.sigma)
    
    def log_prob(self, Y):
        
        return self.forward().log_prob(Y)
    
    def log_prior(self):
        
        return self.mu_prior.log_prob(self.mu) + self.scale_prior.log_prob(self.sigma) 
    
    def train(self, Y, optimizer, n_steps):
    
        losses = np.zeros(n_steps)
        bar = trange(n_steps, leave=True)
        for step in bar:
            optimizer.zero_grad()
            #self.update_params()
            loss = -(self.log_prob(Y).sum() + self.log_prior())
            loss.backward()
            optimizer.step()
            losses[step] = loss
        return losses
    
class LearningGaussianDist_VI(torch.nn.Module):

    def __init__(self):
            super(LearningGaussianDist_VI, self).__init__()
            
            # VI 
                
            # variational params
         
            self.q_mu = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
            self.q_log_sigma = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
                
            self.prior = D.Gamma(torch.tensor([1.0,1.0]), torch.tensor([1.0,1.0]))
            #self.scale_prior = D.Gamma(1.0, 1.0)
    
    def forward(self, mu, sigma):
        
            return D.Normal(mu, sigma)
    
    def log_prob(self, mu, sigma, Y):
        
            return self.forward(mu, sigma).log_prob(Y).sum()
    
    def q(self):
        
          return D.Gamma(self.q_mu, self.q_log_sigma)
        
    def elbo(self, Y): 
        
        loglik = 0.0
        for i in range(50):
            mu, sigma = self.q().rsample()
            loglik += self.log_prob(mu, sigma, Y) 
        
        return loglik/50.0 - kl_divergence(self.q(), self.prior).sum()
       
    def train(self, Y, optimizer, n_steps):
    
            losses = np.zeros(n_steps)
            bar = trange(n_steps, leave=True)
            for step in bar:
                optimizer.zero_grad()
                loss = -self.elbo(Y)
                loss.backward()
                optimizer.step()
                losses[step] = loss
            return losses

if __name__ == '__main__':
    
    # Generate some data
    Y = D.Normal(2,3).sample(torch.tensor([1000]))
    
    lg = LearningGaussianDist()
    lg_vi = LearningGaussianDist_VI()

    optimizer = torch.optim.Adam(lg_vi.parameters(), lr=0.01)

    losses = lg_vi.train(Y, optimizer, 2000)
    