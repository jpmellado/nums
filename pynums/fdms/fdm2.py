#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite-difference approximations to 2. order derivative

import numpy as np

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
    d = np.empty_like(f)
    
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
    d = np.empty_like(f)
    
    # use the python feature that negative index indicates distance to last
    for i in range(n-1):
        d[i] = f[i+1] -2. *f[i] +f[i-1]
    d[n-1] = f[0] -2. *f[n-1] +f[n-2]
    
    return d
