#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:03:44 2019

@author: jpmellado
"""

# Iterative methods to solve linear systems

import numpy as np
import scipy.linalg
import nums.iterative
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text',       usetex=True)
#rc('figure',     dpi=150) # At home, screen 27''
#rc('font',       family='serif', size=10)

n = 20 # size of the matrix

A = np.random.rand(n,n) # Generate random system matrix

# Make it symmetric positive definite for the Gauss-Seidel, steepest descent and conjugate-gradient.
A = np.matmul(A,np.transpose(A)) 
# Make it diagonally dominant for the Jacobi method
# It is already SPD, so we only need to consider row or column
alpha = 1.0 # the larger this is, the smaller the spectral radius and the faster the convergence.
A = A + np.eye(n) *alpha *np.sum(np.abs(A),axis=1)

x = np.random.rand(n)   # Generate random solution
b = np.matmul(A,x)      # Calculate the RHS for that solution
#x = np.ones(n)          # Generate solution equal to ones 
#b = np.sum(A,axis=1)

itmax = n

legends   = [] # Initialize lists

#%% Solve using Jacobi
print('\nSolving system of {} equations by Jacobi.'.format(np.linalg.matrix_rank(A)))
C = np.diag(np.diag(A))
G = np.eye(n)-scipy.linalg.solve(C,A)
_,s,_ = scipy.linalg.svd(G)
print('Spectral radius is {:e}.'.format(np.amax(s)))
x0 = np.zeros((n))
y1,e1 = nums.iterative.Jacobi(A,b,x0,itmax)

legends.append( r"Jacobi, $\rho={:.2f}$".format(np.amax(s)) )

#%% Solve using Gauss-Seidel
print('\nSolving system of {} equations by Gauss-Seidel.'.format(np.linalg.matrix_rank(A)))
C = np.tril(A)
G = np.eye(n)-scipy.linalg.solve(C,A)
_,s,_ = scipy.linalg.svd(G)
print('Spectral radius is {:e}.'.format(np.amax(s)))
x0 = np.zeros((n))
y2,e2 = nums.iterative.GaussSeidel(A,b,x0,itmax)

# Extending the list
legends.append( r"Gauss-Seidel, $\rho={:.2f}$".format(np.amax(s)) )

#%% Solve using steepest descent
print('\nSolving system of {} equations by steepest descent.'.format(np.linalg.matrix_rank(A)))
_,s,_ = scipy.linalg.svd(A)
print('Condition number is {:e}.'.format(np.amax(s) /np.amin(s)))
x0 = np.zeros((n))
y3,e3 = nums.iterative.SteepestDescent(A,b,x0,itmax)

# Extending the list
legends.append( r"Steepest descent, $\kappa={:.2f}$".format(np.amax(s) /np.amin(s)) )

#%% Solve using conjugate gradient
print('\nSolving system of {} equations by conjugate gradient.'.format(np.linalg.matrix_rank(A)))
_,s,_ = scipy.linalg.svd(A)
print('Condition number is {:e}.'.format(np.amax(s) /np.amin(s)))
x0 = np.zeros((n))
y4,e4 = nums.iterative.ConjugateGradient(A,b,x0,itmax)

# Extending the list
legends.append( r"Conjugate gradient, $\kappa={:.2f}$".format(np.amax(s) /np.amin(s)) )

#%% Solve using preconditioned conjugate gradient
print('\nSolving system of {} equations by Jacobi preconditioned conjugate gradient.'.format(np.linalg.matrix_rank(A)))
PA = np.copy(A)
for i in range(np.shape(A)[0]):
    PA[i,:] = PA[i,:] /A[i,i]
_,s,_ = scipy.linalg.svd(PA)
print('Condition number is {:e}.'.format(np.amax(s) /np.amin(s)))
x0 = np.zeros((n))
y5,e5 = nums.iterative.PreconditionedConjugateGradient(A,b,x0,itmax)

# Extending the list
legends.append( r"Preconditioned CG, $\kappa={:.2f}$".format(np.amax(s) /np.amin(s)) )

#%% Plot convergence 

plt.figure( figsize = (4,3)) 
it = np.arange(itmax+1)
plt.plot(it,e1)
plt.plot(it,e2)
plt.plot(it,e3)
plt.plot(it,e4)
plt.plot(it,e5)
plt.legend(legends,loc='best')
plt.yscale("log")
plt.yticks([1e-12,1e-9,1e-6,1e-3,1,1e3])
plt.xlabel(r"Iteration number")
plt.ylabel(r"Residual magnitude")
plt.axis([None, None, 1e-13, 1e4])
plt.tight_layout(pad=0.1)
plt.savefig("iterative01.pdf")
plt.show()