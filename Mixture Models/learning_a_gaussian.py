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
from prettytable import PrettyTable
import pandas as pd

# Bayesian Hierarchical model

class LearningGaussianDist(torch.nn.Module):

    def __init__(self, inference):
            super(LearningGaussianDist, self).__init__()
            
            self.inference = inference
            if self.inference == 'ml':
                
                  self.mu = torch.nn.Parameter(torch.tensor([0.0]))
                  self.sigma = torch.nn.Parameter(torch.tensor([1.0]))
            
            elif self.inference == 'map':
                
                 self.mu = torch.nn.Parameter(torch.tensor([0.0]))
                 self.sigma = torch.nn.Parameter(torch.tensor([1.0]))
                 self.mu_prior = D.Normal(0.0, 1.0)
                 self.scale_prior = D.Gamma(1.0, 1.0)
                 
            else:
                         
                self.q_mu = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
                self.q_log_sigma = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
                
                self.p_mu = torch.tensor([0.0, 0.0])
                self.p_log_sigma = torch.tensor([0.0, 0.0])
                self.prior = D.Normal(self.p_mu, self.p_log_sigma.exp())
                #self.prior = D.Normal(torch.tensor([1.0,1.0]), torch.tensor([1.0,1.0]))
                #self.scale_prior = D.Gamma(1.0, 1.0)
        
    def forward(self, mu=None, sigma=None):
        
            if self.inference == 'vi':
                return D.Normal(mu, sigma)
            else:
                return D.Normal(self.mu, self.sigma)
    
    def log_prob(self, Y, mu=None, sigma=None):
        
            if self.inference == 'vi':
                return self.forward(mu, sigma).log_prob(Y).sum()
            else:
                return self.forward().log_prob(Y).sum()
    
    def log_prior(self):
        
         return self.mu_prior.log_prob(self.mu) + self.scale_prior.log_prob(self.sigma) 
    
    def q(self):
        
          return D.Normal(self.q_mu, self.q_log_sigma.exp())
        
    def elbo(self, Y): 
        
        loglik = 0.0
        for i in range(5):
            mu, log_sigma = self.q().rsample()
            loglik += self.log_prob(Y, mu, log_sigma.exp()) 
        
        return loglik/5.0 - kl_divergence(self.q(), self.prior).sum()
    
    
    def get_trainable_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    def loss_fn(self, Y):
        
        if self.inference == 'map':
            return -(self.log_prob(Y).sum() + self.log_prior())
        elif self.inference == 'ml':
            return -(self.log_prob(Y).sum())
        else: # do fully probabilistic (VI)
            return -self.elbo(Y)
        
    def get_param_names(self):
        
        names = []
        for name, param in self.named_parameters():
            names.append(name)
        return names
    
    def train(self, Y, optimizer, n_steps):
    
            losses = np.zeros(n_steps)
            print('Param states pre-training: ' +  str(self.state_dict()))
            bar = trange(n_steps, leave=True)
            param_states = pd.DataFrame(columns=self.get_param_names())
            param_values = []
            for step in bar:
                optimizer.zero_grad()
                param_values.append(lg.state_dict().values())
                loss = self.loss_fn(Y)
                loss.backward()
                optimizer.step()
                losses[step] = loss
                bar.set_description(str(int(losses[step])))
            print('Param states post-training: ' +  str(self.state_dict()))
            return losses, param_states.from_dict(param_values)
        
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

if __name__ == '__main__':
    
    # Generate some data
    Y = D.Normal(2,3).sample(torch.tensor([1000]))
    
    lg = LearningGaussianDist(inference='vi')

    optimizer = torch.optim.Adam(lg.parameters(), lr=0.01)

    losses, param_values = lg.train(Y, optimizer, 4000)
    







# ML optimisation

# class LearningGaussianDist(torch.nn.Module):
    
#     def __init__(self):
#             super(LearningGaussianDist, self).__init__()
            
#             # Point estimate learning
            
#             self.mu = torch.nn.Parameter(torch.tensor([0.0]))
#             self.sigma = torch.nn.Parameter(torch.tensor([1.0]))
    
#     def forward(self):
        
#         return D.Normal(self.mu, self.sigma)
    
#     def log_prob(self, Y):
        
#         return self.forward().log_prob(Y)
        
#     def train(self, Y, optimizer, n_steps):
    
#         losses = np.zeros(n_steps)
#         bar = trange(n_steps, leave=True)
#         for step in bar:
#             optimizer.zero_grad()
#             #self.update_params()
#             loss = -(self.log_prob(Y).sum())
#             loss.backward()
#             optimizer.step()
#             losses[step] = loss
#         return losses


# # MAP Inference 

# class LearningGaussianDist_MAP(torch.nn.Module):
    
#     def __init__(self):
#             super(LearningGaussianDist_MAP, self).__init__()
            
#             # MAP Inference / Point estimate learning
            
#             self.mu = torch.nn.Parameter(torch.tensor([0.0]))
#             self.sigma = torch.nn.Parameter(torch.tensor([1.0]))
#             self.mu_prior = D.Normal(0.0, 1.0)
#             self.scale_prior = D.Gamma(1.0, 1.0)
    
#     def forward(self):
        
#         return D.Normal(self.mu, self.sigma)
    
#     def log_prob(self, Y):
        
#         return self.forward().log_prob(Y)
    
#     def log_prior(self):
        
#         return self.mu_prior.log_prob(self.mu) + self.scale_prior.log_prob(self.sigma) 
    
#     def train(self, Y, optimizer, n_steps):
    
#         losses = np.zeros(n_steps)
#         bar = trange(n_steps, leave=True)
#         for step in bar:
#             optimizer.zero_grad()
#             loss = -(self.log_prob(Y).sum() + self.log_prior())
#             loss.backward()
#             optimizer.step()
#             losses[step] = loss
#         return losses
    