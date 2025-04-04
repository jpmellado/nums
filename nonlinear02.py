#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:01:02 2019

@author: jpmellado
"""

# Nonlinear systems, 2d functions

import numpy as np
import matplotlib.pyplot as plt
import nums.nonlinear

# In case you want to use latex fonts for the math in the figures
from matplotlib import rc
rc('text',  usetex=True)
rc('font',  size=12)#,    family='serif)

# Define the functions
def g(x):
    n = np.size(x)
    g = np.empty(n)
    g[0] = (x[0] +1.) **2. +x[1] **2. -4.
    g[1] = (x[0] -1.) **2. +x[1] **2. -5.
    return g

def dg(x):
    n = np.size(x)
    dg = np.empty((n,n))
    dg[0,0] = 2. *(x[0]+1.)
    dg[0,1] = 2. * x[1]
    dg[1,0] = 2. *(x[0]-1.)
    dg[1,1] = 2. * x[1]
    return dg

plt.figure( figsize = (6,4)) 
x = np.linspace(-3.5,3.5,50)
y = np.linspace(-3.5,3.5,50)
z = np.empty((50,50,2))
for j in range(np.size(y)):
    for i in range(np.size(x)):
        z[j,i,:] = g([x[i],y[j]])
plt.contour(x,y,z[:,:,0],[0.],colors=['k'])
plt.contour(x,y,z[:,:,1],[0.],colors=['r'])
plt.grid()

itmax = 5

# Initial condition; this determines which root we will get
#x0 =-1.0
x0 = [ 1., 1.]

#%% Look for a first root using Newton-Raphson
y1, e1 = nums.nonlinear.NewtonRaphson(g,dg,x0,itmax)

print("Starting from ({:f},{:f}), the method converges to ({:f},{:f}).".format(x0[0],x0[1],y1[0],y1[1]))
plt.plot(y1[0],y1[1],'o')

# Starting a list with the legends to be used in the plot of convergence at the end
legends = [ "Newton-Raphson" ]

#%% Show the plot that we have been constructing so fat
plt.legend(legends,loc='best')
plt.show()

#%% Plot convergence 

plt.figure( figsize = (6,4)) 
it = np.arange(itmax+1)

plt.plot(it,e1)
plt.legend(legends,loc='best')
plt.yscale("log")
plt.xlabel(r"Iteration number")
plt.ylabel(r"Residual magnitude")
plt.tight_layout(pad=0.1)
plt.savefig("nonlinear03.pdf")
plt.show()