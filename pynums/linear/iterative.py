#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:41:55 2019

@author: jpmellado
"""

import numpy as np
import scipy.linalg

def Jacobi(A,b,x0,itmax):
# Solves the system Ax=b for x using steepest descent method
# Input parameters
#    - A: System matrix 
#    - b: Right-hand side vector
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    e[0] = scipy.linalg.norm(np.matmul(A,x)-b)          # error estimate for initial guess
    
    for it in range(itmax):
        y = np.copy(x) # temporary array containing previous approximation
        for i in range(n):
            x[i] = ( b[i] - np.dot(A[i,:i],y[:i]) -np.dot(A[i,i+1:],y[i+1:]) )/ A[i,i]
            
        e[it+1] = scipy.linalg.norm(np.matmul(A,x)-b)   # error estimate in column space
        #print('Norm of residual at iteration number {} is {:e}'.format(it+1,e[it]))
    
    return x, e

def GaussSeidel(A,b,x0,itmax):
# Solves the system Ax=b for x using steepest descent method
# Input parameters
#    - A: System matrix 
#    - b: Right-hand side vector
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(x0)
    e = np.zeros(itmax+1)    
    e[0] = scipy.linalg.norm(np.matmul(A,x)-b)          # error estimate for initial guess
    
    for it in range(itmax):
        for i in range(n):
            x[i] = ( b[i] - np.dot(A[i,:i],x[:i]) -np.dot(A[i,i+1:],x[i+1:]) )/ A[i,i]
            
        e[it+1] = scipy.linalg.norm(np.matmul(A,x)-b)   # error estimate in column space
        #print('Norm of residual at iteration number {} is {:e}'.format(it+1,e[it]))
    
    return x, e

def SteepestDescent(A,b,x0,itmax):
# Solves the system Ax=b for x using steepest descent method
# Input parameters
#    - A: System matrix 
#    - b: Right-hand side vector
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    # Initialize
    r = np.matmul(A,x) -b                   # No need to distinguish between gradient r and descent direction
    e[0] = scipy.linalg.norm(r)             # error estimate for initial guess
    
    # Obtain the remaining approximations reusing the value of Ar
    for it in range(itmax):
        Ar = np.matmul(A,r)
        alpha =-np.dot(r,r) /np.dot(r,Ar)
        x = x +alpha *r                     # update position
        #e[it] = scipy.linalg.norm(alpha*r)  # error estimate in origin space
        r = r +alpha *Ar                    # update residual (gradient direction)

        e[it+1] = scipy.linalg.norm(r)      # error estimate in column space        
        # print('Residual at iteration number {} is {:e}'.format(it+1,e[it]))
    
    return x, e

def ConjugateGradient(A,b,x0,itmax):
# Solves the system Ax=b for x using conjugate gradient method
# Input parameters
#    - A: System matrix 
#    - b: Right-hand side vector
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    # Initialize
    r = np.matmul(A,x) -b                   # gradient vector
    p =-np.copy(r)                          # descent direction
    e[0] = np.dot(r,r)                      # The error estimate is part of the algorithm
    
    # Obtain the remaining approximations reusing the value of Ar
    for it in range(itmax):
        Ap = np.matmul(A,p)
        alpha = e[it] /np.dot(p,Ap)
        x = x +alpha *p                     # update position
        #e[it] = scipy.linalg.norm(alpha*d)  # error estimate in origin space
        r = r +alpha *Ap                    # update residual (gradient direction)
        e[it+1] = np.dot(r,r)
        beta = e[it+1] / e[it]
        p =-r + beta *p                     # obtain new conjugate direction
    
    e = np.sqrt(e)      # L2-norm of error estimate in column space
            
    return x, e

def PreconditionedConjugateGradient(A,b,x0,itmax):
# Solves the system Ax=b for x using conjugate gradient method
# Input parameters
#    - A: System matrix 
#    - b: Right-hand side vector
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    # Initialize
    r = np.matmul(A,x) -b
    s = np.divide(r,np.diag(A))
    p =-np.copy(s)
    e[0] = np.dot(r,s)  # The error estimate is part of the algorithm
    
    # Obtain the remaining approximations reusing the value of Ar
    for it in range(itmax):
        Ap = np.matmul(A,p)
        alpha = e[it] /np.dot(p,Ap)
        x = x +alpha *p                     # update position
        #e[it] = scipy.linalg.norm(alpha*d)  # error estimate in origin space
        r = r +alpha *Ap                    # update residual (gradient direction)
        s = np.divide(r,np.diag(A))
        e[it+1] = np.dot(r,s)
        beta = e[it+1] / e[it]
        p =-s + beta *p                     # obtain new conjugate direction
    
    e = np.sqrt(e)      # L2-norm of error estimate in column space
            
    return x, e

