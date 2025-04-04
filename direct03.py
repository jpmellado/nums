#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:30:59 2019

@author: jpmellado
"""

# Direct methods: Checking fill-in of factorization methods

import numpy as np
import matplotlib.pyplot as plt
import nums.direct

n = 20 # size of the matrix

def plot_fillin(A,L,U,file):
    fig, ((f1,f2,f3)) = plt.subplots(1,3,figsize=(8,3))
    plt.subplot(f1)
    plt.spy(A) # plot sparsity
    plt.title('System Matrix',y=-.15)
    plt.subplot(f2)
    plt.spy(L) # plot sparsity
    plt.title('Lower triangular',y=-.15)
    plt.subplot(f3)
    plt.spy(U, precision=1e-15) # plot sparsity; see round-off errors
    plt.title('Upper triangular',y=-.15)
    plt.tight_layout(pad=1,h_pad=1,w_pad=1)
    plt.savefig("{}.pdf".format(file))
    plt.show()

#%% Example 1
A = np.diag(np.random.rand(n)) # Generate random system matrix
A[0,:] = np.random.rand(n)
A[:,0] = np.random.rand(n)

L, U = nums.direct.LUFactDoolittle(A)
plot_fillin(A,L,U,'fillin1')

#%% Example 2
A = np.diag(np.random.rand(n)) # Generate random system matrix
A[n-1,:] = np.random.rand(n)
A[:,n-1] = np.random.rand(n)

L, U = nums.direct.LUFactDoolittle(A)
plot_fillin(A,L,U,'fillin2')

#%% Example 3
A = np.diag(np.random.rand(n)) # Generate random system matrix
A[int(n/2),:] = np.random.rand(n)
A[:,int(n/2)] = np.random.rand(n)

L, U = nums.direct.LUFactDoolittle(A)
plot_fillin(A,L,U,'fillin3')
