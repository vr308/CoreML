#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:45:46 2020

@author: vr308
"""

from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)

x = torch.rand(5, 3)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)


x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)               

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions


x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2 

z = y * y * 3
out = z.mean()

print(z, out)

out.backward()

x = torch.tensor(1.0, requires_grad = True)
z = x ** 3

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2