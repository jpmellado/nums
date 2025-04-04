#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:30:07 2019

@author: jpmellado
"""

# Validating Thomas algorithm

import numpy as np
import scipy.linalg
import nums.direct
import matplotlib.pyplot as plt

n = 10

a = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(n)

# Construct a full arrayfor validation with general routines
A = np.diagflat(b) + np.diagflat(a[1:],-1) + np.diagflat(c[:n-1],1)

# Construct solution and right-hand side
x = np.random.rand(n)
d = np.matmul(A,x) # Calculate the RHS for that solution

#%% LU factorization
nums.direct.LUFActThomasInPlace(a,b,c)

# Checking fill-in of circulant matrices
#A[0,n-1] = np.random.rand(1)
#A[n-1,0] = np.random.rand(1)
#
#nums.direct.LUFactDoolittleInPlace(A)
#plt.spy(np.tril(A))
#plt.show()
#
#plt.spy(np.triu(A))
#plt.show()

#%% Solving the system
y = nums.direct.ThomasSolve(a,b,c,d)
print('Error {:e}.'.format(np.linalg.norm(x-y)))

y = scipy.linalg.solve(A,d)
print('Error {:e}.'.format(np.linalg.norm(x-y)))
