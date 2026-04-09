#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite-difference approximations

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from template import *
import nums.fdm

# Figure names and counter
tag = 'derivatives'
fig_id = 0

#################################################
# Define functions
def exp_(x):
    return (                    # Parenthesis to write comments at end of line
            'Exp',              # Name
            np.exp(x),          # Function
            np.exp(x),          # First-order derivative
            np.exp(x)           # Second-order derivative
            )      

def sin_(x):
    return 'Sin', \
           np.sin(np.pi *x), \
           np.cos(np.pi *x) *np.pi, \
          -np.sin(np.pi *x)*np.pi **2.

def cos_(x):
    return 'Cos', \
           np.cos(np.pi *x), \
          -np.sin(np.pi *x) *np.pi, \
          -np.cos(np.pi *x) *np.pi **2.

def gauss_(x):
    c = 20.
    tmp = np.exp(-c *x *x)
    return 'Gaussian', \
           tmp, \
          -tmp *2. *c *x, \
          -tmp *2. *c *( 1. -2. *c *x *x )

# Create grid
n = 20 
x = np.linspace(-1.,1.,n)
h = (x[n-1]-x[0]) /(n-1)

# Define list of functions to be processed
fs = [ exp_(x), sin_(x), cos_(x), gauss_(x)]

#################################################
#%% Calculate FD approximation to the first-order derivative
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

[ plt.plot(x, f[2], label=f[0]) for f in fs ]

plt.gca().set_prop_cycle(None)
fdm1 = [ nums.fdm.fdm1_e121(f[1]) /h for f in fs ]
[ plt.plot(x, y, 'o') for y in fdm1 ]

plt.title("First-order derivative")
plt.xlabel("$x$")
plt.ylabel("$df/dx$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
#%% Calculate FD approximation to the second-order derivative
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

[ plt.plot(x, f[3], label=f[0]) for f in fs ]

plt.gca().set_prop_cycle(None)
fdm2 = [ nums.fdm.fdm2_e121(f[1]) /h**2. for f in fs ]
[ plt.plot(x, y, 'o') for y in fdm2 ]

plt.title("Second-order derivative")
plt.xlabel("$x$")
plt.ylabel("$d^2f/dx^2$")
plt.legend(loc="best")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
#%% Convergence study: we increment the number of grid points n by factors of 2
# between 2**imin and 2**imax
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

h   = []
e1s = []
e2s = []

for i in range(4,11):
    n = 2 **i
    
    x = np.linspace(-1.,1.,n)
    h.append( (x[n-1]-x[0]) /(n-1) )
        
    # Define list of functions to be processed
    fs = [ exp_(x), sin_(x), cos_(x), gauss_(x)]
    # Calculate FD approximation to the first-order derivative and error
    fdm1s = [ nums.fdm.fdm1_e121(f[1]) /h[-1] for f in fs ]
    e1s.append( [ scipy.linalg.norm(fdm1s[i]-fs[i][2]) /np.sqrt(float(n)) for i in range(len(fdm1s)) ] )
#    e1s.append( [ np.amax(np.abs(fdm1s[i]-fs[i][2])) for i in range(len(fdm1s)) ] )
    # Calculate FD approximation to the second-order derivative
    fdm2s = [ nums.fdm.fdm2_e121(f[1]) /h[-1]**2. for f in fs ]
    e2s.append( [ scipy.linalg.norm(fdm2s[i]-fs[i][3]) /np.sqrt(float(n)) for i in range(len(fdm2s)) ] )
#    e2s.append( [ np.amax(np.abs(fdm2s[i]-fs[i][3])) for i in range(len(fdm2s)) ] )

    # Add one case with periodic boundary conditions
    f = cos_(x[:-1])
    fdm1 = nums.fdm.fdm1_e2p(f[1]) /h[-1]
    e1s[-1] += [ scipy.linalg.norm(fdm1-f[2]) /np.sqrt(float(n)) ]
#    e1s[-1] += [ np.amax(np.abs(fdm1-f[2])) ]
    fdm2 = nums.fdm.fdm2_e2p(f[1]) /h[-1] **2.0
    e2s[-1] += [ scipy.linalg.norm(fdm2-f[3]) /np.sqrt(float(n)) ]
#    e2s[-1] += [ np.amax(np.abs(fdm2-f[3])) ]

    legends = [ f[0] for f in fs ] + ['Cos periodic']

h_ = np.array( h )          # We need arrays to plot

e1s_ = np.array( e1s )      # We need arrays to plot
for i in range(np.shape(e1s_)[1]):
    plt.plot( h_/h_[0], e1s_[:,i] /e1s_[0,i] )
plt.legend(legends,loc='best')
plt.xscale("log")
plt.yscale("log")
plt.title("First-order derivative")
plt.xlabel("Grid spacing $h/h_0$")
plt.ylabel("Global error $e_2/e_{2,0}$")
#plt.ylabel("Global error $e_\infty/e_{\infty,0}$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

e2s_ = np.array( e2s )      # We need arrays to plot

for i in range(np.shape(e1s_)[1]):
    plt.plot( h_/h_[0], e2s_[:,i] /e2s_[0,i] )
plt.legend(legends,loc='best')
plt.xscale("log")
plt.yscale("log")
plt.title(r"Second-order derivative")
plt.xlabel(r"Grid spacing $h/h_0$")
plt.ylabel(r"Global error $e_2/e_{2,0}$")
#plt.ylabel(r"Global error $e_\infty/e_{\infty,0}$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
#%% Plots for the slides
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )

x=np.linspace(0,np.pi,100)

plt.plot(x,x)
plt.plot(x,np.sin(x))
plt.legend([r"Exact",r"FD approximation"],loc='best')
plt.title(r"First-order derivative")
plt.xlabel(r"Scaled wavenumber $w_k=(2\pi/L)kh=(2\pi/n)k$")
plt.ylabel(r"modified wavenumber $\tilde{w}_k$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
plt.show()
