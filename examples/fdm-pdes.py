#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Examples of Finite Difference Methods for Advection and Diffusion Equations

import numpy as np
import matplotlib.pyplot as plt
from template import *

# Figure names and counter
tag = 'fdm-pdes'
fig_id = 0

###############################################################################
# Pre-processing: Define the input data

# Define the spatial grid, uniformly spaced
xmin = -1.
xmax = 1.
nx = 20
x = np.linspace(xmin, xmax, nx)
h = (x[nx-1]-x[0]) /(nx-1)

# Define the problem
diffusivity = None      # Default
velocity = None         # Default
# diffusivity = 0.2
velocity = 1.0
from nums.pdes.advection import *

# Define the temporal grid
tmin = 0.               # Initial time
tmax = 8.0              # Final time
dt_variable = True
if dt_variable:         # Variable time step according to stability constraint
    dnum = 0.2          # Diffusion number: Euler < 2./4., RK3 < 2.57/4.
    cnum = 2.0          # CFL number: RK3 < 1.73
else:                   # Constant time step
    dt = 0.021052632            # Time step

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
else:
    if diffusivity != None:
        dnum = dt*diffusivity/h**2.
    if velocity != None:
        cnum = dt*velocity/h

# from nums.pdes.timemarching import Euler as timescheme
from nums.pdes.timemarching import RungeKutta3 as timescheme

# Checkpointing meta-data (save a set of times); using python lists
check_dt = 2.0                  # Interval at which data is saved

###############################################################################
# Running simulation

# Checkpointing meta-data (save a set of times); using python lists 
check_tsaved = []               # space for time
check_usaved = []               # space for the numerical solution

# Initialize loop
t = tmin                        # Time
u = ics(x)

check_tsaved.append(t)          # Save initial condition
check_usaved.append(np.copy(u))
check_t = t +check_dt           # Next time for checkpointing
it = 0                          # To keep track of how many iterations we need
  
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
        
###############################################################################
# Post-processing: Plot functions

fig_id = fig_id +1
plt.figure( figsize = figsize11 )

cmap = plt.get_cmap('magma_r')
legends = []

for item in range(len(check_tsaved)):
    legends.append(r"Time ${:5.2f}$".format(check_tsaved[item]))
    factor = float(item+1)/float(len(check_tsaved)+1)
    plt.plot(x,check_usaved[item],color='darkred',alpha=factor)
#    plt.plot(x,check_usaved[item],color=cmap(factor))

legends.append(r'Reference')
plt.plot(x,reference(u,t),color='black')

if diffusivity != None:
    plt.title(r"Diffusion \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(dnum,nx,it))
if velocity != None:
    plt.title(r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(cnum,nx,it))
    
plt.xlabel(r"position $x$")
plt.ylabel(r"function $u$")
#plt.legend(legends,bbox_to_anchor=(1.02,0.5), loc="center left")
plt.legend(legends,loc='best')
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

plt.show()


