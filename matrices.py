#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:10:58 2019

@author: jpmellado
"""

import scipy.io
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
import time

path = './data/'
# 2D Elliptic problem (SPD matrix)
# Looks like a 52x52 grid with Dirichlet BCs 
# and we solve for 50x50 interior values
file1 = ['matrices_51.mat', 
        'matrices_101.mat', 
        'matrices_201.mat']

# 2D convection-diffusion problem
# Looks like 51x51 grid where (x,y) contains 50+50+50+50 boundary nodes
# and we solve for 51*51 values + 50*4 boundary values
file2 = ['matnosim_51.mat', # 
        'matnosim_101.mat',
        'matnosim_201.mat']

data = scipy.io.loadmat(path+file2[2])

x = data['x']
y = data['y']

A0 = data['A0'].astype(float)
#A0_= csc_matrix(A0).toarray()
b0 = data['b0']
numer0 = data['numer0']

A1 = data['A1'].astype(float)
#A1_= csc_matrix(A1).toarray()
b1 = data['b1']
numer1 = data['numer1']

#%% Banded case
print("Ordered node numbering:")
n = int(np.sqrt(np.size(b0)))
#m = int(np.sqrt(np.size(b0)-np.size(x)))
# Make it positive definite
A0  = -A0
#A0_ = -A0_
b0  = -b0

#plt.spy(A0)
#plt.show()

#t0 = time.process_time()
#LU, piv = scipy.linalg.lu_factor(A0_,check_finite=False)
#sol0 = scipy.linalg.lu_solve((LU,piv),b0)
#print('Time using Gauss {:e}.'.format(time.process_time()-t0))
#
#t0 = time.process_time()
#C, low = scipy.linalg.cho_factor(A0_,lower=True,check_finite=False)
#sol0 = scipy.linalg.cho_solve((C,low),b0)
#print('Time using Cholesky {:e}.'.format(time.process_time()-t0))

t0 = time.process_time()
sol0 = scipy.sparse.linalg.spsolve(A0,b0)
print('Time using sparsity {:e}.'.format(time.process_time()-t0))

t0 = time.process_time()
sol0 = scipy.sparse.linalg.cg(A0,b0,tol=1e-2)
print('Time using sparsity and CG {:e}.'.format(time.process_time()-t0))

#plt.contourf(sol0.reshape(n,n))
##plt.contourf(sol0[:m*m].reshape(m,m))
#plt.colorbar()
#plt.axis('equal')
#plt.show()

#%% Non-banded case
print("\nRandom node numbering:")
n = int(np.sqrt(np.size(b1)))
#m = int(np.sqrt(np.size(b1)-np.size(x)))
A1  = -A1
#A1_ = -A1_
b1  = -b1

#plt.spy(A1)
#plt.show()

#t0 = time.process_time()
#LU, piv = scipy.linalg.lu_factor(A1_,check_finite=False)
#sol1 = scipy.linalg.lu_solve((LU,piv),b1)
#print('Time using Gauss {:e}.'.format(time.process_time()-t0))
#
#t0 = time.process_time()
#C, low = scipy.linalg.cho_factor(A1_,lower=True,check_finite=False)
#sol1 = scipy.linalg.cho_solve((C,low),b1)
#print('Time using Cholesky {:e}.'.format(time.process_time()-t0))

t0 = time.process_time()
sol1 = scipy.sparse.linalg.spsolve(A1,b1)
print('Time using sparsity {:e}.'.format(time.process_time()-t0))

t0 = time.process_time()
sol1 = scipy.sparse.linalg.cg(A1,b1,tol=1e-2)
print('Time using sparsity and CG {:e}.'.format(time.process_time()-t0))

#tmp = np.zeros(np.size(sol1))
#for i in range(0,np.size(sol1)):
#    tmp[numer1[i]-1] = sol1[i]
#plt.contourf(tmp.reshape(n,n))
##plt.contourf(tmp[:m*m].reshape(m,m))
#plt.colorbar()
#plt.axis('equal')
#plt.show()
