#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:51:15 2019

@author: jpmellado
"""

# Nonlinear systems, scalar functions

import numpy as np
import matplotlib.pyplot as plt
import nums.nonlinear

# In case you want to use latex fonts for the math in the figures
from matplotlib import rc
rc('text',  usetex=True)
rc('font',  size=12)#,    family='serif)

# Maximum number of iterations
itmax = 6

# Define the functions
def g(x):
    g = 0.25 *x *(x-1.) *(x+2.) 
    return g

def dg(x):
    dg = 0.25 *( (x-1.) *(x+2.) +x *(x+2.) +x *(x-1.) )
    return dg

# Initialize list for common legends in plots
legends = []

# Initialize the figure of the function and the roots
plt.figure( figsize = (4,3)) 
x = np.linspace(-2.5,1.5,50)
plt.plot( x, g(x), color='k', label='function' )
plt.grid()

# Initial condition; this determines which root we will get
x0 =-0.75
#x0 = 1.25
plt.plot( x0, g(x0), 'o', color='k' )

#%% Look for a second root using Bisection
x1 = 0.5 # needs two values to initialize the method with a sign-change between them
y1, e1 = nums.nonlinear.Bisection(g,x0,x1,itmax-1)

print( "Starting from {:f}, the bisection method converges to {:f}".format(x0,y1) )
plt.plot( y1, g(y1), 'o' )
legends.append( "Bisection" )

#%% Look for a first root using Successive approximations
y2, e2 = nums.nonlinear.SuccessiveApproximations(g,x0,itmax)

print("Starting from {:f}, the fixed-point method converges to {:f}".format(x0,y2))
plt.plot( y2, g(y2), '^' )
legends.append( "Fixed-point" )

#%% Look for a first root using Newton-Raphson
y3, e3 = nums.nonlinear.NewtonRaphson1D(g,dg,x0,itmax)

print("Starting from {:f}, the Newton method converges to {:f}".format(x0,y3))
plt.plot( y3, g(y3), 'v' )
legends.append( "Newton" )

#%% Look for a first root using secant method
x1 = x0 +0.25
y4, e4 = nums.nonlinear.Secant1D(g,x0,x1,itmax-1)

print("Starting from {:f}, the secant method converges to {:f}".format(x0,y4))
plt.plot( y4, g(y4), 'x' )
legends.append( "Secant" )

#%% Show the plot that we have been constructing so far
plt.legend(['function','initial guess']+legends,loc='best',ncol=2) #['function','initial guess']+legends,loc='best', ncol=2)
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.tight_layout(pad=0.1)
plt.savefig("nonlinear01.pdf")
plt.show()

#%% Plot convergence 

plt.figure( figsize = (4,3)) 
it = np.arange(itmax+1)

plt.plot(it,e1 /e1[0])
plt.plot(it,e2 /e2[0])
plt.plot(it,e3 /e3[0])
plt.plot(it,e4 /e4[0])

plt.axis([None, None, 1e-16 , 1e1 ])
plt.grid()
plt.legend(legends,loc='best')
plt.yscale("log")
plt.xlabel(r"Iteration number")
plt.ylabel(r"Residual magnitude")
plt.tight_layout(pad=0.1)
plt.savefig("nonlinear02.pdf")
plt.show()