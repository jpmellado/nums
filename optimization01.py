#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:09:13 2019

@author: jpmellado
"""

# Optimization, scalar functions

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# In case you want to use latex fonts for the math in the figures
from matplotlib import rc
rc('text',  usetex=True)
rc('font',  size=12)#,    family='serif)

# Define the functions
def g(x):
    g = x *x *(x-1.) *(x+2.)
    return g

def dg(x):
    dg = (x-1.) *(x+2.) +x *(x+2.) +x *(x-1.)
    return dg

plt.figure( figsize = (4,3)) 
x = np.linspace(-2.5,1.5,50)
plt.plot(x,g(x),color='k')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')

#%% Finding local minima
res1 = scipy.optimize.minimize_scalar(g,bounds=(-2.,-1.0),method='bounded')
print('First minimum {:e}.'.format(res1.x))
plt.plot(res1.x,g(res1.x),'o')

res2 = scipy.optimize.minimize_scalar(g,bounds=(0.5,1.0),method='bounded')
print('First minimum {:e}.'.format(res2.x))
plt.plot(res2.x,g(res2.x),'o')

plt.tight_layout(pad=0.1)
plt.savefig("optimization01.pdf")
plt.show()