#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:59:54 2020
@author: vr308
Variational Auto-Encoder in pytorch
"""

import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

float_tensor = lambda x: torch.tensor(x, dtype=torch.float)

def get_batch(x, size):
    
    N = len(x)
    valid_indices = np.array(range(N))
    batch_indices = np.random.choice(valid_indices,size=size,replace=False)
    return float_tensor(x[batch_indices,:])

features = 16

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.Linear = torch.nn.Linear(D_in, H)
        self.Linear = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        
        x = F.relu(self.Linear(x))
        return 

class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            # encoder
            self.encoder = nn.Sequential(
                    nn.Linear(in_features=784, out_features=512),
                    nn.Linear(in_features=512, out_features=features*2))
            # decoder
            self.decoder = nn.Sequential(
                nn.Linear(in_features=features, out_features=512),
                nn.Linear(in_features=512, out_features=784))
            
        def forward(self, x):
            
            # encoding
            x = F.relu(self.encoder(x)).reshape(-1,2, features)
            
            # get mu and log_var
            mu = x[:,0,:]
            log_var = x[:,1,:]
            
            # get the latent representation
            std = torch.sqrt(torch.exp(log_var))
            eps = torch.randn_like(std)
            z = mu + eps*std
            
            #decoding
            x = F.relu(self.decoder(z))
            reconstruction = torch.sigmoid(x)
            
            return reconstruction, mu, log_var
        
        @staticmethod
        def get_batch(x, batch_size):
            
            N = len(x)
            valid_indices = np.array(range(N))
            batch_indices = np.random.choice(valid_indices,size=batch_size,replace=False)
            return float_tensor(x[batch_indices,:])
        
        def loss_fn(self, x, reconstruction, z_mu, z_logvar):
            
            bce_loss = nn.BCELoss(reduction='sum')
            q_z = torch.distributions.Normal(z_mu, torch.sqrt(torch.exp(z_logvar)))
            p_z = torch.distributions.Normal(torch.tensor([0.0]*features), torch.tensor([1.0]*features))
            kl_div = -torch.distributions.kl_divergence(q_z,p_z)
            final_loss = bce_loss(reconstruction, x) - kl_div
            return final_loss.mean()
        
        def fit(self, x, batch_size, optimizer, n_epochs):
            trace_loss = []
            for i in range(n_epochs):
                x_batch = get_batch(x, batch_size)
                optimizer.zero_grad()
                reconstruction, mu, log_var = self.forward(x_batch)
                loss = self.loss_fn(x_batch, reconstruction, mu, log_var)
                loss.backward()
                optimizer.step()
                trace_loss.append(loss)
                if i%1000 == 0:
                    print('Loss at step {}: {}'.format(i,loss.item()))
            return trace_loss

        def reconstruct(self, z_mu):
            
            return;

def mean_squared_error(X_true, X_reconstruction):
    return;
    

def negative_log_likelihood(X_true, X_reconstruction):
    return;            
            


if __name__ == '__main__':

      from tensorflow.keras.datasets.mnist import load_data
    
      (y_train, train_labels), (y_test, test_labels) = load_data()
      labels = np.hstack([train_labels, test_labels])
      n = len(labels)
      Y = np.vstack([y_train, y_test])
      Y = float_tensor(Y.reshape(n, -1)/255.0)
    
      # learning parameters
      learning_rate = 0.001
      batch_size = 200
      epochs = 5000
      
      model = LinearVAE()
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      
      trace_loss = model.fit(Y, batch_size, optimizer, 5000)