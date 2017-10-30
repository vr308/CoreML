# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

x = np.sort(np.random.uniform(-5,5,1000))
y = np.sort(np.random.uniform(-5,5,1000))

def linear_kernel(x,y,c):
    
    return x*y + c
    
def periodic(x,y,l):
    
    return np.exp(-2*np.square(np.sin((x-y)/2))/np.square(l))

def squared_exponential(x,l):
    
    return np.exp(-np.square(x-0)/(2*np.square(l)))

def rational_quadratic(x,y, alpha):
    
    return np.power((1 + np.power(np.abs(x-y),2)),-2)

def brownian_kernel(x,y):
    
    return 



