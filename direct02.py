#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:30:59 2019

@author: jpmellado
"""

# Direct methods: Checking times

import numpy as np
import scipy.linalg
import time

n = 10 # size of the matrix

A = np.random.rand(n,n) # Generate random system matrix

A = np.matmul(A,np.transpose(A)) # Make it symmetric positive definite

#A = A + np.eye(n) *np.sum(np.abs(A),axis=1) # Make it diagonally dominant

#x = np.random.rand(n)   # Generate random solution
#b = np.matmul(A,x)      # Calculate the RHS for that solution
x = np.ones(n)          # Generate solution equal to ones 
b = np.sum(A,axis=1)

#%% Comparing execution times
print('Solving system of {} equations.'.format(np.linalg.matrix_rank(A)))

t0 = time.process_time()
#LU = scipy.linalg.lu(A,check_finite=False)
LU, piv = scipy.linalg.lu_factor(A,check_finite=False)
sol0 = scipy.linalg.lu_solve((LU,piv),b)
print('Time using Gauss {:e}.'.format(time.process_time()-t0))

t0 = time.process_time()
#C = scipy.linalg.cholesky(A,lower=True,check_finite=False)
C, low = scipy.linalg.cho_factor(A,lower=True,check_finite=False)
sol0 = scipy.linalg.cho_solve((C,low),b)
print('Time using Cholesky {:e}.'.format(time.process_time()-t0))
