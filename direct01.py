#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:30:59 2019

@author: jpmellado
"""

# Direct methods for linear systems: LU Factorizations and solutions

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time
import nums.direct

n = 10 # size of the matrix

A = np.random.rand(n,n) # Generate random system matrix
x = np.random.rand(n)   # Generate random solution

A = np.matmul(A,np.transpose(A)) # make it symmetric positive definite

AD = np.diag(A) # Define a diagonal matrix
AU = np.triu(A) # Define an upper tridiagonal matrix
AL = np.tril(A) # Define a lower tridiagonal matrix

#%% Solve the lower triangular system
print('Solving lower triangular system of {} equations.'.format(np.linalg.matrix_rank(AL)))
b = np.matmul(AL,x) # Calculate the RHS for that solution

t0 = time.process_time()
y = nums.direct.LTriSolve(AL,b)
print('Time {:e}.'.format(time.process_time()-t0))
print('Error {:e}.'.format(np.linalg.norm(x-y)))

t0 = time.process_time()
y = scipy.linalg.solve_triangular(AL,b,lower=True)
print('Time using SciPy {:e}.'.format(time.process_time()-t0))
print('Error {:e}.\n'.format(scipy.linalg.norm(x-y)))

#%% Solve the upper triangular system
print('Solving upper triangular system of {} equations.'.format(np.linalg.matrix_rank(AU)))
b = np.matmul(AU,x) # Calculate the RHS for that solution

t0 = time.process_time()
y = nums.direct.UTriSolve(AU,b)
print('Time {:e}.'.format(time.process_time()-t0))
print('Error {:e}.'.format(scipy.linalg.norm(x-y)))

t0 = time.process_time()
y = scipy.linalg.solve_triangular(AU,b)
print('Time using SciPy {:e}'.format(time.process_time()-t0))
print('Error using SciPy {:e}\n'.format(scipy.linalg.norm(x-y)))

#%% Solve full system
print('Solving full system of {} equations.'.format(np.linalg.matrix_rank(A)))
b = np.matmul(A,x) # Calculate the RHS for that solution

t0 = time.process_time()
y,_ = nums.direct.GaussSolve(A,b)
print('Time {:e}.'.format(time.process_time()-t0))
print('Error {:e}.'.format(scipy.linalg.norm(x-y)))

#t0 = time.process_time()
#nums4.direct.GaussSolveInPlace(A,b)
#print('Time {:e}.'.format(time.process_time()-t0))
#print('Error {:e}.'.format(scipy.linalg.norm(x-b)))

t0 = time.process_time()
y = scipy.linalg.solve(A,b)
print('Time using SciPy {:e}'.format(time.process_time()-t0))
print('Error using SciPy {:e}.\n'.format(scipy.linalg.norm(x-y)))

#%% Calculate Cholesky factorization
print('Calculate Cholesky factorization.')

t0 = time.process_time()
L = nums.direct.LFactCholesky(A)
print('Time {:e}.'.format(time.process_time()-t0))
print('Error {:e}.'.format(scipy.linalg.norm(A - np.matmul(L,np.transpose(L)))))

t0 = time.process_time()
L = scipy.linalg.cholesky(A,lower=True,check_finite=False)
print('Time using Scipy {:e}.'.format(time.process_time()-t0))
print('Error using SciPy {:e}.\n'.format(scipy.linalg.norm(A - np.matmul(L,np.transpose(L)))))

#t0 = time.process_time()
#nums4.direct.LFactCholeskyInPlace(A)
#print('Time {:e}.'.format(time.process_time()-t0))
#print('Error {:e}.'.format(scipy.linalg.norm(np.tril(A) - L)))

#%% Calculate LU factorization of A
print('Calculate LU decomposition.')

t0 = time.process_time()
P, L, U = nums.direct.PLUFactDoolittle(A)
print('Time {:e}.'.format(time.process_time()-t0))
print('Error {:e}.'.format(scipy.linalg.norm(np.matmul(P,A)-np.matmul(L,U))))

t0 = time.process_time()
P, L, U = scipy.linalg.lu(A,check_finite=False)
print('Time using SciPy {:e}.'.format(time.process_time()-t0))
print('Error using SciPy {:e}.\n'.format(scipy.linalg.norm(A-np.matmul(P,np.matmul(L,U)))))

#P1 = nums4.direct.PLUFactDoolittleInPlace(A)
#U1 = np.triu(A)
#L1 = np.tril(A)

fig, ((f1,f2)) = plt.subplots(1,2)
plt.subplot(f1)
plt.spy(L) # plot sparsity
plt.title('Lower triangular',y=-.15)
plt.subplot(f2)
plt.spy(U, precision=1e-15) # plot sparsity; see round-off errors
plt.title('Upper triangular',y=-.15)
plt.show()


