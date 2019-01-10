#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:50:42 2018

@author: vr308
"""

import numpy as np
from scipy import random
import timeit
import matplotlib.pylab as plt
import scipy.optimize as so

# Generate a random positive definite square matrix of size N x N

def gen_pos_def_matrix(N):
    
    A = random.rand(N,N)
    return np.matrix(np.dot(A, A.T))

import time

def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % round(end_ts - beg_ts, 4) + " sec.")
        return retval
    return wrapper

@time_usage
def get_inverse(K):
    return np.linalg.inv(K)

def cube(x, k):
    
    return k*x*x*x

def woodbury_matrix_identity(A, U, C, V):
    
    A_inv = np.linalg.inv(A)
    A_inv_U = np.matmul(A_inv, U)
    C_inv = np.linalg.inv(C)
    V_A_inv_U = np.matmul(np.matmul(V, A_inv),U)
    V_A_inv = np.matmul(V, A_inv)
    return A_inv - np.matmul(np.matmul(A_inv_U, np.linalg.inv(C_inv + V_A_inv_U)),V_A_inv)

@time_usage
def block_matrix_inversion(K):
    
    A, B, C, D = border_decomp_matrix(K)
    A_inv = np.linalg.inv(A)
    schur_com = schur_complement_inverse(A,B,C,D)
    inv_11 = A_inv + A_inv*B*schur_com*C*A_inv
    inv_12 = -A_inv*B*schur_com
    inv_21 = -schur_com*C*A_inv
    inv_22 = schur_com
    return np.bmat([[inv_11, inv_12],[inv_21, inv_22]])
    
def border_decomp_matrix(K):
    
    dim = K.shape[0]
    K_11 = K[0:dim-1,0:dim-1]
    K_12 = K[0:dim-1, dim-1]
    K_21 = K[dim-1, 0:dim-1]
    K_22 = np.matrix(K[dim-1, dim-1])
    return K_11, K_12, K_21, K_22 
    
def schur_complement_inverse(A,B,C,D):
    
    return woodbury_matrix_identity(D, -C, np.linalg.inv(A), B)
    
    
if __name__ == "__main__":

         lags = []
         fast_lags = []
         dims = [10,100,500,1000,1500, 2000, 3000, 5000]
         for i in dims:
         
             K = gen_pos_def_matrix(i)
             start1 = timeit.default_timer()
             get_inverse(K)
             lags.append(timeit.default_timer() - start1)
             
             start2 = timeit.default_timer()
             block_matrix_inversion(K)
             fast_lags.append(timeit.default_timer() - start2)


# Show that the time complexity is actually O(N^3)

popt, pcov = so.curve_fit(cube, dims,fast_lags)

plt.figure()
plt.plot(dims, fast_lags/popt)
plt.plot(dims, dims*np.square(dims))

# Updating matrix inversion using Woodbury Matrix identity

A = np.matrix(data= [[2,3,1],[1,4,1],[2,3,4]])
A_inv = np.linalg.inv(A)

B = np.matrix(data= [[1,0,1],[1,3,0],[0,0,4]])
B_inv = np.linalg.inv(B)

AB_inv = np.linalg.inv(A + B)
AB_inv_check = A_inv - np.matmul(np.matmul(A_inv, np.linalg.inv(B_inv + A_inv)),A_inv)

# Using partition into block matrices and the woodbury identity to expand the schur complement 

A_11 = A[0:2,0:2]
A_12 = A[0:2,2]
A_21 = A[2,0:2]
A_22 = np.matrix(A[2,2])

F_11 = A_11 - np.matmul(np.matmul(A_12,np.linalg.inv(A_22)), A_21)
F_22 = A_22 - np.matmul(np.matmul(A_21,np.linalg.inv(A_11)), A_12)

F_11_inv = woodbury_matrix_identity(A_11, -A_12, np.linalg.inv(A_22), A_21)
F_22_inv = woodbury_matrix_identity(A_22, -A_21, np.linalg.inv(A_11), A_12)

A_11_inv = np.linalg.inv(A_11)
A_22_inv = np.linalg.inv(A_22)

A_inv_check = np.bmat([[F_11_inv, -A_11_inv*A_12*F_22_inv], 
                       [-F_22_inv*A_21*A_11_inv, F_22_inv]])

 





