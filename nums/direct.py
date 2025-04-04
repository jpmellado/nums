#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:46:25 2019

@author: jpmellado
"""
import numpy as np

def LTriSolve(A,b):
# Solves the Lower Triangular System Ax=b for x
# Input parameters
#    - A: System matrix (assumed lower triangular)
#    - b: Right-hand side vector
# Output parameters
#    - x: Solution vector
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    x = np.copy(b)
    x[0] = x[0] /A[0,0]
    for k in range(1,n):
        x[k] = ( x[k] - np.dot(A[k,:k],x[:k]) )/A[k,k]
        
    return x

def UTriSolve(A,b):
# Solves the Upper Triangular System Ax=b for x
# Input parameters
#    - A: System matrix (assumed upper triangular)
#    - b: Right-hand side vector
# Output parameters
#    - x: Solution vector
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 

    x = np.copy(b)
    x[n-1] = x[n-1] /A[n-1,n-1]
    for k in range(n-2,-1,-1):  
        x[k] = ( x[k] - np.dot(A[k,k+1:],x[k+1:]) )/A[k,k]

    return x

def GaussSolve(A,b):
# Doolittle LU factorization (row operations)
# InPlace operation to minimize memory requirements    
# Input objects 
#   - A: System matrix
#   - b: Right-hand side vector    
# Output objects
#   - x: Solution vector    
#   - LU: lower and upper triangular matrix    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    LU= np.copy(A)
    x = np.copy(b)
    
    for k in range(0,n-1):
        LU[k+1:,k] = LU[k+1:,k] /LU[k,k] # Update k column of L with multipliers
        LU[k+1:,k+1:] = LU[k+1:,k+1:] -np.outer( LU[k+1:,k] ,LU[k,k+1:] ) # Upate k minor of U
        x[k+1:] = x[k+1:] - LU[k+1:,k] *x[k] # Extend row operations to the RHS
        
    x[n-1] = x[n-1] /LU[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = ( x[k] - np.dot(LU[k,k+1:],x[k+1:]) )/LU[k,k]
        
    return x, LU

def GaussSolveInPlace(A,b):
# Doolittle LU factorization (row operations)
# InPlace operation to minimize memory requirements    
# Input objects 
#   - A: System matrix
#   - b: Right-hand side vector    
# Output objects
#   - A: lower and upper triangular matrix
#   - b: Solution vector    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    for k in range(0,n-1):
        A[k+1:,k] = A[k+1:,k] /A[k,k] # Update k column of L with multipliers
        A[k+1:,k+1:] = A[k+1:,k+1:] -np.outer( A[k+1:,k] ,A[k,k+1:] ) # Upate k minor of U
        b[k+1:] = b[k+1:] - A[k+1:,k] *b[k] # Extend row operations to the RHS
        
    b[n-1] = b[n-1] /A[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = ( b[k] - np.dot(A[k,k+1:],b[k+1:]) )/A[k,k]
    
def LUFactDoolittle(A):
# Doolittle LU factorization (row operations)
# Input objects 
#   - A: System matrix
# Output objects
#   - L: lower triangular matrix
#   - U: upper triangular matrix    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    L = np.eye(n)  # initialize as indentity matrix
    U = np.copy(A) # out-of-place operation to keep A
    for k in range(0,n-1):
        L[k+1:,k] = U[k+1:,k] /U[k,k] # Update k column of L with multipliers
        U[k+1:,k:] = U[k+1:,k:] -np.outer( L[k+1:,k] ,U[k,k:] ) # Upate k minor of U
        
    return L, U

def PLUFactDoolittle(A):
# Doolittle LU factorization (row operations) with partial pivoting PA = LU
# Input objects 
#   - A: System matrix
# Output objects
#   - P: permutation matrix    
#   - L: lower triangular matrix
#   - U: upper triangular matrix    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    P = np.eye(n)  # initialize as identity matrix
    L = np.eye(n)  # initialize as identity matrix
    U = np.copy(A) # out-of-place operation to keep A
    for k in range(0,n-1):
        kmax = k +(np.abs(U[k:,k])).argmax() # find the index of max in the column
        if k != kmax: # swap current row for pivot row in P, L and U using advanced slicing
            P[[k,kmax]    ] = P[[kmax,k]    ] # swap the permutation
            U[[k,kmax],k: ] = U[[kmax,k],k: ] # swap the U, i.e., calculate PA
            if k > 0:
                L[[k,kmax], :k] = L[[kmax,k], :k] # swap the L, i.e., calculate PL
        L[k+1:,k] = U[k+1:,k] /U[k,k] # Update k column of L with multipliers
        U[k+1:,k:] = U[k+1:,k:] -np.outer( L[k+1:,k] ,U[k,k:] ) # Upate k minor of U
        
    return P, L, U

def LUFactDoolittleInPlace(A):
# Doolittle LU factorization (row operations)
# InPlace operation to minimize memory requirements    
# Input objects 
#   - A: System matrix
# Output objects
#   - A: lower and upper triangular matrix
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    for k in range(0,n-1):
        A[k+1:,k] = A[k+1:,k] /A[k,k] # Update k column of L with multipliers
        A[k+1:,k+1:] = A[k+1:,k+1:] -np.outer( A[k+1:,k] ,A[k,k+1:] ) # Upate k minor of U
        
def PLUFactDoolittleInPlace(A):
# Doolittle LU factorization (row operations) with partial pivoting PA = LU
# InPlace operation to minimize memory requirements    
# Input objects 
#   - A: System matrix
# Output objects
#   - P: permutation matrix    
#   - A: lower and upper triangular matrix
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    P = np.eye(n)  # initialize as identity matrix
    for k in range(0,n-1):
        kmax = k +(np.abs(A[k:,k])).argmax() # find the index of max in the column
        if k != kmax: # swap current row for pivot row in P, L and U using advanced slicing
            P[[k,kmax] ] = P[[kmax,k] ] # swap the permutation
            A[[k,kmax] ] = A[[kmax,k] ] # swap the U, i.e., calculate PA (last m-k+1 columns)
                                        # and swap the L, i.e., calculate PL (first k-1 colums)
        A[k+1:,k] = A[k+1:,k] /A[k,k] # Update k column of L with multipliers
        A[k+1:,k+1:] = A[k+1:,k+1:] -np.outer( A[k+1:,k] ,A[k,k+1:] ) # Upate k minor of U
        
    return P

def LFactCholesky(A):
# Cholesky LL' factorization
# Input parameters
#    - A: System matrix
# Output parameters
#    - L: Lower triangular matrix    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    L = np.tril(A)
    
    # First block
    if A[0,0] <= 0.:
        print('Error: Matrix is not positive definite')
        return
    L[0,0] = np.sqrt(A[0,0])
    
    # Remaining blocks
    for k in range(0,n-1):
        if A[k+1,k+1] <= 0.:
            print('Error: Matrix is not positive definite')
            return
        
        # Substitution step; solving lower-triangular system of size k+1 for L[k+1,0:k+1]
        L[k+1,0] = A[k+1,0] /L[0,0]
        for j in range(1,k+1):
            L[k+1,j] = ( A[k+1,j] - np.dot(L[j,:j],L[k+1,:j]) )/L[j,j]
            
        # Diagonal term    
        L[k+1,k+1] = np.sqrt( A[k+1,k+1] - np.dot(L[k+1,:k+1],L[k+1,:k+1]))
        
    return L

def LFactCholeskyInPlace(A):
# Cholesky LL' factorization
# Input parameters
#    - A: System matrix
# Output parameters
#    - A: Lower triangular matrix    
    n, m = np.shape(A) # number of rows znd columns
    if m != n:
        print('Error: Matrix is not square.')
        return 
    
    # First block
    if A[0,0] <= 0.:
        print('Error: Matrix is not positive definite')
        return
    A[0,0] = np.sqrt(A[0,0])
    
    # Remaining blocks
    for k in range(0,n-1):
        if A[k+1,k+1] <= 0.:
            print('Error: Matrix is not positive definite')
            return
        
        # Substitution step; solving lower-triangular system of size k+1 for L[k+1,0:k+1]
        A[k+1,0] = A[k+1,0] /A[0,0]
        for j in range(1,k+1):
            A[k+1,j] = ( A[k+1,j] - np.dot(A[j,:j],A[k+1,:j]) )/A[j,j]
            
        # Diagonal term    
        A[k+1,k+1] = np.sqrt( A[k+1,k+1] - np.dot(A[k+1,:k+1],A[k+1,:k+1]))
        
def LUFActThomasInPlace(a,b,c):
# LU factorization of a tridiagonal matrix
# Input parameters
#    - a: vector containing the sub-diagonal; a[0] not used
#    - b: vector containing the diagonal
#    - c: vector containing the super-diagonal: c[n-1] not used
# Output parameters
#    - a: vector containing the sub-diagonal of L; a[0] not used
#    - b: vector containing the diagonal of U
#    - c: vector containing the super-diagonal of U: c[n-1] not used    
    n = np.size(a)
    if np.size(b) != n:
        print('Error: vectors a and b have different size')
    if np.size(c) != n:
        print('Error: vectors a and d have different size')
        
    for i in range(1,n):
        a[i] = a[i] /b[i-1]
        b[i] = b[i] - a[i] *c[i-1]

def ThomasSolve(a,b,c,d):
# Substitution step in Thomas algorithm    
# Input parameters
#    - a: vector containing the sub-diagonal of L; a[0] not used
#    - b: vector containing the diagonal of U
#    - c: vector containing the super-diagonal of U: c[n-1] not used
#    - d: right-hand side of linear system
    n = np.size(a)
    if np.size(b) != n:
        print('Error: vectors a and b have different size')
    if np.size(c) != n:
        print('Error: vectors a and c have different size')
    if np.size(d) != n:
        print('Error: vectors a and d have different size')
    
    x = np.copy(d)
    
# Forward substitution step
    for i in range(1,n):
        x[i] = x[i] -a[i] *x[i-1]
        
# Backward substitution step
    x[n-1] = x[n-1] /b[n-1]
    for i in range(n-2,-1,-1):
        x[i]= ( x[i] -c[i] *x[i+1] ) /b[i]
    
    return x