#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:58:06 2020

@author: vr308
"""
import torch

# Linear layers

#1d

x = torch.randn(10) # 1x10
lin_layer = torch.nn.Linear(in_features=10,out_features=20)
output = lin_layer(x) # 20x1

weight = lin_layer.weight # 20x10
bias = lin_layer.bias # 20x1
manual = torch.matmul(x, weight.T) + bias

assert((manual - output).sum() == 0.0)

# 2d 

x = torch.randn(10,2)
lin_layer = torch.nn.Linear(in_features=2,out_features=7)
output = lin_layer(x) # 10x7

weight = lin_layer.weight # 7x2
bias = lin_layer.bias # 7x1
manual = torch.matmul(x, weight.T) + bias

assert((manual - output).sum() == 0.0)

# Convolutional layers

# 1d
x = torch.randn(1, 3, 5)
conv1d = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, stride=1)
output = conv1d(x)

#2d

x = torch.rand(100, 10, 28,28)
conv2d = torch.nn.Conv2d(10,32,2,stride=1)
weight = conv2d.weight
bias = conv2d.bias
output = conv2d(x)

# Avg/max pooling

x = torch.randn(20, 16, 50)
max1d = torch.nn.MaxPool1d(2, stride=1)
max1d(x)

x = torch.randn(10,1,3,3)
max2d = torch.nn.MaxPool2d(2, stride=1)
max2d(x)

#Avg pooling is the same, just replace max with avg

# Recurrent layer

rnn = torch.nn.RNNCell(10,20, bias=False, nonlinearity='relu')
input = torch.randn(3,10)
output = rnn(input)

# Batch Norm

#1d

x = torch.randn(40,100)
batch_norm = torch.nn.BatchNorm1d(100) 
output = batch_norm(x)

means = torch.mean(x, axis=0)
sd = torch.std(x, unbiased=False, axis=0)

manual = (x - means)/sd

assert((manual - output).sum() < 1e-5)


#2d

x = torch.randn(40, 3, 10, 10)
batch_norm2d = torch.nn.BatchNorm2d(3, affine=False) 
output = batch_norm2d(x)

means = torch.mean(x, axis=(0,2,3))
sd = torch.std(x, unbiased=False, axis=(0,2,3))

manual = (x - means[None,:,None,None])/sd[None,:,None,None]

assert((manual - output).abs().max() < 1e-4)


# Activations -> elemwise, wont change shape

x = torch.randn(10)

#sigmoid 

sig = torch.nn.Sigmoid()
output = sig(x)
manual = 1/(1 + torch.exp(-x))

print(manual - output)


#relu

relu = torch.nn.ReLU()
relu(x)

#tanh

tanh = torch.nn.Tanh()
tanh(x)


#softmax

#1d

x = torch.randn(10,50)
softmax = torch.nn.Softmax(dim=0)
output = softmax(x)
output[:,0].sum() == 1.0

#2d

x = torch.randn(40, 3, 10, 10)
softmax2d = torch.nn.Softmax2d()
output = softmax2d(x)
#on the 2nd dim
output[0][:,0,0].sum()

# Attention (1 head)

import math

  # Inputs:
  #       - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
  #         the embedding dimension.
  #       - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
  #         the embedding dimension.
  #       - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
  #         the embedding dimension.
  
    # Outputs:
    #    - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
    #      E is the embedding dimension.
    #    - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
    #      L is the target sequence length, S is the source sequence length.

attn = torch.nn.MultiheadAttention(10,1, bias=False)

query = torch.randn(5,100,10)
values = torch.randn(2,100,10)
keys = torch.randn(2,100,10)

attn_output, attn_o_weights = attn(query, keys, values)
attn_o_weights[0]

proj_weight = attn.in_proj_weight

q = torch.matmul(query, proj_weight[0:10,:].T)
k = torch.matmul(keys, proj_weight[10:20,:].T)
v = torch.matmul(values, proj_weight[20:30,:].T)

presoft = torch.matmul(q.permute(1,0,2), k.permute(0,2,1).T)/math.sqrt(10)

softmax = torch.nn.Softmax(dim=-1)
postsoft = softmax(presoft)

postsoft[0]

manual_output = torch.matmul(postsoft,v.permute(1,0,2))
final_output = torch.matmul(manual_output, attn.out_proj.weight.T) + attn.out_proj.bias

attn_output[0][0]
final_output[0][0]

assert ((attn_output[0][0] - final_output[0][0]).sum() < 1e-5)