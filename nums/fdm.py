#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite-difference approximations

import numpy as np

def fdm1_e121(f):
# Calculates the FD approximation to the first-order derivative in uniform grids
# using a explicit formulation with biased formulas at the boundaries.
# It is second-order in the interior points, first order at the boundaries.
# Still need to divide by the grid spacing h.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    d[0] = f[1] -f[0]
    for i in range(1,n-1):
        d[i] = 0.5 *(f[i+1]-f[i-1])
    d[n-1] = f[n-1]-f[n-2]

    return d

def fdm1_e2p(f):
# Calculates the FD approximation to the first-order derivative in uniform grids
# using a explicit formulation with periodic boundary conditions.
# The last point in the array is one before the end of the periodic interval.
# It is second-order.
# Still need to divide by the grid spacing h.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    # use the python feature that negative index indicates distance to last
    for i in range(n-1):
        d[i] = 0.5 *(f[i+1]-f[i-1])
    d[n-1] = 0.5 *(f[0]-f[n-2])
    
    return d

def fdm1_e11(f):
# Calculates the FD approximation to the first-order derivative in uniform grids
# using a explicit formulation with forward biased formulas
# Still need to divide by the grid spacing h.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    for i in range(0,n-1):
        d[i] = f[i+1]-f[i]
    d[n-1] = f[n-1]-f[n-2]

    return d

def fdm1_e11(f):
# Calculates the FD approximation to the first-order derivative in uniform grids
# using a explicit formulation with forward biased formulas
# Still need to divide by the grid spacing h.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    for i in range(n-1):
        d[i] = f[i+1]-f[i]
    d[n-1] = f[n-1]-f[n-2]

    return d

def fdm1_e1p(f):
# Same, but periodic boundary conditions
# The last point in the array is one before the end of the periodic interval.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    for i in range(n-1):
        d[i] = f[i+1]-f[i]
    d[n-1] = f[0]-f[n-1]

    return d

def fdm2_e121(f):
# Calculates the FD approximation to the second-order derivative in uniform grids
# using a explicit formulation with biased formulas at the boundaries.
# It is second-order in the interior points, first order at the boundaries.
# Still need to divide by the grid spacing h^2.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    d[0] = f[2] -2. *f[1] +f[0]
    for i in range(1,n-1):
        d[i] = f[i+1] -2. *f[i] +f[i-1]
    d[n-1] = f[n-1] -2.* f[n-2] +f[n-3]

    return d

def fdm2_e2p(f):
# Calculates the FD approximation to the second-order derivative in uniform grids
# using a explicit formulation with periodic boundary conditions.
# It is second-order.
# Still need to divide by the grid spacing h^2.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    
    # use the python feature that negative index indicates distance to last
    for i in range(n-1):
        d[i] = f[i+1] -2. *f[i] +f[i-1]
    d[n-1] = f[0] -2. *f[n-1] +f[n-2]
    
    return d

def fdm1_c6_lhs(n):
# Creates the system matrix for the FD approximation to the first-order in uniform grids
# using 6th-order compact schemes.
# Input arguments:
#    - n: number of grid points
# Output arguments:
#    - a,b,c: vectors with the diagonals of the system matrix
    dummy = 1. /3.
    a = dummy *np.ones(n)
    b =        np.ones(n)
    c = dummy *np.ones(n)

    c[0] = 2.
    a[1] = 1. /6.
    c[1] = 1. /2.

    a[n-2] = 1. /2.
    c[n-2] = 1. /6.
    a[n-1] = 2.

    return a, b, c
    
def fdm1_c6_rhs(f):
# Right-hand side corresponding to the left-hand side above.
# Still need to divide by the grid spacing h.
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: vector with the right-hand side of the system
    n = np.size(f)
    d = np.empty(n)
    
    d[0] = -5./2. *f[0] +2.0 *f[1] +0.5 *f[2]
    d[1] = -5./9. *f[0] -0.5 *f[1] +     f[2] +1./18. *f[3]
    for i in range(2,n-2):
        d[i] = 7./9. *(f[i+1]-f[i-1]) +1./36. *(f[i+2]-f[i-2])
    d[n-2] =-1./18. *f[n-4] -     f[n-3] +0.5 *f[n-2] +5./9. *f[n-1]
    d[n-1] =                -0.5 *f[n-3] -2.0 *f[n-2] +5./2. *f[n-1]

    return d

