#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:30:26 2017

@author:  vr308
"""

# Simple 2-layer Neural network 

import numpy as np

# sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(1000):

    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1,True)

    # update weights
    syn0 = syn0 + np.dot(l0.T,l1_delta)
    
print('Output After Training:')
print(l1)

# 3-Layer Neural Network

X = np.array([[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]])
    
y = np.array([[0],[1], [1], [0]])

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for iter in xrange(40000):
    
    #Feed forward through the layers 0,1 and 2
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    
    # how much did we miss ?
    l2_error = y - l2
    
    if (iter% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # 
    l2_delta = l2_error*sigmoid(l2,deriv=True)
    
    # How much l1 contributed to the l2 error
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*sigmoid(l1,deriv=True)
    
    # Weight update rule - comes out of the calculus
    syn1 = syn1 + l1.T.dot(l2_delta)
    syn0 = syn0 + l0.T.dot(l1_delta)

print "Output after Training"
print l2


###################
# MLP using Python libraries
####################

import sklearn.datasets as skdata
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from random import shuffle

scaler = StandardScaler()
data, targets = skdata.load_wine(return_X_y=True)

index = [i for i in range(len(data))]
shuffle(index)

X_train = scaler.fit_transform(data[index][0:100:,])
y_train = targets[index][0:100]
X_test = scaler.fit_transform(data[index][100:,])
y_test = targets[index][100:]

net = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', activation='relu')
trained_net = net.fit(X_train, y_train)
y_pred = trained_net.predict(X_test)

test_accuracy = net.score(X_test, y_test)
print(np.round(test_accuracy*100,2))

# Cross-checking

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

test_data = X_test

# Collecting the weights

layer_01_weights = net.coefs_[0]
layer_12_weights = net.coefs_[1]

# Layer 1 outputs 

relu = lambda x: x*(x > 0) 
activations_l1 = relu(np.dot(test_data, layer_01_weights) + net.intercepts_[0])

# Layer 2 outputs

activations_l2 = softmax(np.dot(activations_l1,layer_12_weights) + net.intercepts_[1])

def map_to_class(outputs):
    
    return outputs.index(max(outputs))
    
y_pred1 = []
for i in range(len(activations_l2)):
    y_pred1.append(map_to_class(list(activations_l2[i]))) 
    
(y_pred == y_pred1).all()