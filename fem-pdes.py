#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Examples of Finite Element Methods for Diffusion Equations

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import nums.fdm

# In case we want to use latex fonts in matplotlib
from matplotlib import rc
rc('text',  usetex=True)
rc('font',  family='serif', size=11)
rc('axes',  titlesize='medium')

#
# Pre-processing: Define the input data
#

# Define the case
def rhs(u,t):           # Right-hand side of equation
    return RhsDiffusionDirichlet(u,t)
def bcs(u,t):           # Boundary conditions
    return BcsDiffusionDirichlet(u,t)
def ics(x):             # Initial conditions
    return IcsDiffusionDirichlet(x)
def reference(u,t):     # Reference function to compare with
    return RefDiffusionDirichlet(u,t)
diffusivity = 0.2
#diffusivity = None
velocity = None
#velocity = 1.0

def timescheme(f,u,t):
    return Euler(f,u,t)
#    return RungeKutta3(f,u,t)

# Define the spatial grid
ixmax = 20
x = np.linspace(-1.,1.,ixmax)   # Uniformly spaced between -1. and 1.
h = (x[ixmax-1]-x[0]) /(ixmax-1)

# Define the temporal grid
tmin = 0.               # Initial time
tmax = 8.0              # Final time
dt_variable = True
if dt_variable:         # Variable time step according to stability constraint
    dnum = 0.1          # Diffusion number: Euler < 2./4., RK3 < 2.57/4.
    cnum = 0.2          # CFL number: RK3 < 1.73
else:                   # Constant time step
    dt = 0.1            # Time step

# Checkpointing meta-data (save a set of times); using python lists
check_dt = 2.0
#
# Running simulation
# Code below should not be modified
#

# Calculate time step, if necessary
if dt_variable:
    if diffusivity != None:
        dtd = dnum *(h **2.) /diffusivity
    if velocity != None:
        dtc = cnum *h /velocity
    if diffusivity != None and velocity != None:
        dt = min(dtd,dtc)
    elif diffusivity != None and velocity == None:
        dt = dtd
    else: 
        dt = dtc

# Define the functions
def RhsDiffusionDirichlet(u,t):                 # Right-hand side of evolution equation (tendency)
    n = np.size(x)
    f = np.ones(n)                              # Source term: constant
    f = diffusivity *nums.fdm.fdm2_e121(u) /( h **2. ) +fil_e141(f)
    A = np.empty((3,n-2))                       # Tridiagonal mass matrix
    A[0,:] = 1. /6.
    A[1,:] = 4. /6.
    A[2,:] = 1. /6.
    f[1:n-1] = scipy.linalg.solve_banded((1,1),A,f[1:n-1])
    # Calculate boundary conditions; Dirichlet
    # Initial condition sets the Dirichlet conditions
    f[ 0] = 0. # Include first and last node in state vector
    f[-1] = 0. # but keep them fixed
    return f

def RefDiffusionDirichlet(u,t):                 # Reference function to compare with
# Exact solution for steady diffusion equation with Dirichlet boundary conditions
    f = -0.5 *(x-1.) *(x+1.) /diffusivity       # constant source term
    slope = ( u[ixmax-1] -u[0] ) /( x[ixmax-1] -x[0] )
    f = f +x[0] +slope *( x +1. )               # linear profile between boundaries
    return f

def BcsDiffusionDirichlet(u,t):                 # Boundary conditions, nothing to be done
    return

def IcsDiffusionDirichlet(x):                   # Initial conditions
    f = np.empty(np.size(x))
    f[:] = x
    return f

def fil_e141(f):
# Calculates the top-hat filter for 2h with a Simpson rule
# Input arguments:
#    - f: values of the function at the grid points
# Output arguments:
#    - d: approximations to the derivative of the function at the grid points 
    n = np.size(f)
    d = np.empty(n)
    c = 1./6.
    
    d[0] = c *( 2. *f[0] +f[1] )
    for i in range(1,n-1):
        d[i] = c *( f[i+1] +4.*f[i] +f[i-1] )
    d[n-1] = c *( 2. *f[n-1] +f[n-2] )

    return d

# Define time-marching schemes
def Euler(rhs,u,t):
# rhs: function that calculates the rhs of the evoluton equation
# u: vector with state variables
# t: time
    f = u +dt *rhs(u,t)   
    return f, t +dt

def RungeKutta3(rhs,u,t):   
# rhs: function that calculates the rhs of the evoluton equation
# u: vector with state variables
# t: time
    k1 = rhs( u,                     t          )
    k2 = rhs( u +dt *0.5 *k1,        t +0.5 *dt )
    k3 = rhs( u +dt *( 2. *k2 -k1 ), t +     dt )
    f = u +dt *( k1 +4. *k2 +k3 ) /6.
    return f, t +dt
   
# Checkpointing meta-data (save a set of times); using python lists 
check_tsaved = []            # space for time
check_usaved = []            # space for the numerical solution

# Initialize loop
t = tmin                # Time
u = ics(x)

check_tsaved.append(t)       # Save initial condition
check_usaved.append(np.copy(u))
check_t = t +check_dt   # Next time for checkpointing
it = 0                  # To keep track of how many iterations we need
  
# Integration loop
while t < tmax:
    u, t = timescheme(rhs,u,t)  # Advance one time step
    bcs(u,t)                    # Corrections to satisfy the boundary conditions, if needed
    it = it +1
    # Saving data 
    if t >= check_t:
        print("Checking data at t = {:5.2f} after {:3d} iterations.".format(t,it))
        check_tsaved.append(t)
        check_usaved.append(np.copy(u))
        check_t = check_t +check_dt

# Always save the last time, if not done
if t >= check_t:
    print("Checking data at t = {:5.2f} after {:3d} iterations.".format(t,it))
    check_tsaved.append(t)
    check_usaved.append(np.copy(u))
        
#    
# Post-processing: Plot functions
#
plt.figure( figsize = (4,3))
legends = []
cmap = plt.cm.get_cmap('magma_r')
for item in range(len(check_tsaved)):
    legends.append(r"Time ${:5.2f}$".format(check_tsaved[item]))
    factor = float(item+1)/float(len(check_tsaved)+1)
    plt.plot(x,check_usaved[item],color='darkred',alpha=factor)
plt.plot(x,reference(u,t),color='black')
legends.append(r'Reference')
#plt.legend(legends,bbox_to_anchor=(1.02,0.5), loc="center left")
plt.legend(legends,loc='best')
if diffusivity != None:
    plt.title(r"Diffusion \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(dnum,ixmax,it))
if velocity != None:
    plt.title(r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(cnum,ixmax,it))
plt.xlabel(r"position $x$")
plt.ylabel(r"function $u$")
plt.grid()
plt.tight_layout(pad=0.1)
plt.savefig("fem1.pdf",bbox_inches='tight')
plt.show()


