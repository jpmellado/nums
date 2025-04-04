#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:45:34 2019

@author: jpmellado
"""

# Optimization of functions of 2D variables

import numpy as np
import matplotlib.pyplot as plt
import nums.optimize

# In case you want to use latex fonts for the math in the figures
from matplotlib import rc
rc('text',       usetex=True)
rc('font',       family='serif', size=12)

# Define various objective functions

## First function is one parabolla with one minimum at (1,0)
#def g(x):
#    g = 3. *(x[0] -1.) **2. +    x[1] **2. -5.
#    return g
#
#def dg(x):      # gradient
#    dg = np.empty(np.size(x))
#    dg[0] = 2. *( x[0] -1.)
#    dg[1] = 6. *x[1]
#    return dg
#
#def d2g(x):     # Hessian
#    n = np.size(x)
#    d2g = np.zeros((n,n))
#    d2g[0,0] = 2.
#    d2g[1,1] = 6.
#    return d2g

# Second function is two parabolla with two minima at (-1,0) and (1,0)
c1, c2, c3, c4 = 4.0, 1.0, 0.5, 0.2
def g(x):
    a =      x[0]      **2. +    x[1] **2. -c4
    b = c1 *(x[0] -1.) **2. +c2 *x[1] **2. -c3
    g = a *b
    return g

def dg(x):      # gradient
    dg = np.empty(np.size(x))
    a =      x[0]      **2. +    x[1] **2. -c4
    b = c1 *(x[0] -1.) **2. +c2 *x[1] **2. -c3
    dg[0] = 2. *x[0] *b + 2. *c1 *( x[0] -1.) *a
    dg[1] = 2. *x[1] *b + 2. *c2 *x[1]        *a
    return dg

def d2g(x):     # Hessian
    n = np.size(x)
    d2g = np.zeros((n,n))
    a =      x[0]      **2. +    x[1] **2. -c4
    b = c1 *(x[0] -1.) **2. +c2 *x[1] **2. -c3
    d2g[0,0] = 2. *b + 2. *c1 *a + 8. *c1 *x[0] *(x[0] -1.) 
    d2g[1,1] = 2. *b + 2. *c2 *a + 8. *c2 *x[1] **2.
    d2g[0,1] = 4. *x[0] *c2 *x[1] + 4. *c1 *( x[0] -1.) *x[1]
    d2g[1,0] = d2g[0,1]
    return d2g

## Third function is one parabolla along a parabolla with one minimum at (1,1)
#c1 = 10.
#
#def g(x):
#    g = c1 *(x[1]-x[0] **2.) **2. +(1-x[0]) **2.
#    return g
#
#def dg(x):      # gradient
#    dg = np.empty(np.size(x))
#    a = c1 *( x[1] -x[0] **2. )
#    dg[0] =-4. *a *x[0] -2. *(1-x[0])
#    dg[1] = 2. *a 
#    return dg
#
#def d2g(x):     # Hessian
#    n = np.size(x)
#    d2g = np.zeros((n,n))
#    a = c1 *( x[1] -x[0] **2. )
#    d2g[0,0] =-4. *a +8. *c1 *x[0] **2. +2. 
#    d2g[1,1] = 2. *c1
#    d2g[0,1] =-4. *c1 *x[0]
#    d2g[1,0] = d2g[0,1]
#    return d2g

# Plot the function
x1 = np.linspace(-0.5, 1.5 ,50)
x2 = np.linspace(-1.0, 1.0 ,50)
x1g, x2g = np.meshgrid(x1, x2)
x = np.array([x1g,x2g])
z = g(x)

plt.figure( figsize = (8,6))
ax = plt.axes(projection="3d")
ax.plot_surface(x1g, x2g, z, cmap='viridis', alpha=0.5)
ax.contour(x1g, x2g, z, 20, colors=['k'])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

plt.figure( figsize = (4,3))
#plt.pcolormesh(x, y, z, shading='gouraud', vmin = z.min(), vmax = 0.1 *z.max() )
plt.contour(x1, x2, z, 20)#, colors='k')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid()

itmax = 10

# Initial condition; this determines which minima we will get
x0 = [ 0.7, 0.6]

legends = []

#%% Optimize using steeoest descent
y1,e1 = nums.optimize.SteepestDescent(g,dg,x0,itmax)

print("Starting from ({:f},{:f}), the method converges to ({:f},{:f}).".format(x0[0],x0[1],y1[0],y1[1]))
plt.plot(y1[0],y1[1],'o')

legends.append( "Steepest Descent" )

#%% Optimize using conjugate-gradient (Fletcher-Reeves)
y2,e2 = nums.optimize.FletcherReeves(g,dg,x0,itmax)

print("Starting from ({:f},{:f}), the method converges to ({:f},{:f}).".format(x0[0],x0[1],y2[0],y2[1]))
plt.plot(y2[0],y2[1],'o')

legends.append( "Fletcher-Reeves" )

#%% Optimize using modified Newton
y3,e3 = nums.optimize.Newton(g,dg,d2g,x0,itmax)

print("Starting from ({:f},{:f}), the method converges to ({:f},{:f}).".format(x0[0],x0[1],y3[0],y3[1]))
plt.plot(y3[0],y3[1],'o')

legends.append( "Modified Newton" )

#%% Show the plot that we have been constructing so far
plt.plot(x0[0],x0[1],marker='o',color='k') # Plot initial condition

plt.tight_layout(pad=0.1)
plt.savefig("optimization02.pdf")
plt.show()

#%% Plot convergence

plt.figure( figsize = (4,3))
it = np.arange(itmax+1)

plt.plot(it,e1)
plt.plot(it,e2)
plt.plot(it,e3)
plt.legend(legends,loc='best')
plt.yscale("log")
plt.xlabel(r"Iteration number")
plt.ylabel(r"Residual magnitude")
plt.tight_layout(pad=0.1)
plt.savefig("optimization03.pdf")
plt.show()
