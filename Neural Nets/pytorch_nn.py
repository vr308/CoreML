#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:40:22 2020

@author: vr308
"""

import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(), 
    torch.nn.Linear(H, D_out)
    )

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate=1e-4
for t in range(500):
    
    y_pred = model(x)
    
    loss = loss_fn(y_pred, y)
    
    model.zero_grad()
    
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            print(param)
            break;
            param -= learning_rate*param.grad