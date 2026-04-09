#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Compare different time-marching schemes

import numpy as np
import matplotlib.pyplot as plt
import nums.ode
from template import *

# define the function and the normalized solution
def f(u,t):
    return -u

def sol(t):
    return np.exp(-t)
 
# Grid properties
t0 = 0.                     # Initial time
tn = 2. *np.pi              # Final time

u0 = 1.                     # Initial condition

# Exact solution
te = np.linspace( t0, tn, 50 )   # Increase resolution to show it smoother
ue = u0 *sol(te) 

# Figure names and counter
tag = 'ode'
fig_id = 0

#################################################
#%% 1 ode
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )
plt.title(r"Equation $u'=-u$")
plt.plot( te, ue, label=r'exact' )

# Grid properties
n  = 15                     # Number of steps
dt = ( tn -t0 )/ n          # Step size

# Numerical solution using various schemes
legends =[]                                     # Create list for the schemes
schemes =[]
legends.append( "Euler" )
schemes.append( nums.ode.Euler )
legends.append( "Adams-Bashforth 2" )
schemes.append( nums.ode.AdamsBashforth2 )
legends.append( "Runge-Kutta 2" )
schemes.append( nums.ode.RungeKutta2 )
# legends.append( "Runge-Kutta 3" )
# schemes.append( nums.ode.RungeKutta3 )

u = [ s(f,t0,u0,dt,n) for s in schemes ]        # Integrate 

for i in range(len(schemes)):                   # Plot
    plt.plot(u[i][0], u[i][1], 'o', label=legends[i] )
#    plt.plot(u[i][0], u[i][1][:,0], 'o', label=legends[i] )

plt.legend(loc='best')
plt.xlabel(r"time $t$")
plt.ylabel(r"function $u$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
# system of 1 ode; stability
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )
plt.title(r"Equation $u'=-u$. Euler $(\Delta t)_\mathrm{max}=2.0$")
plt.plot(te, ue, label='exact')

ns = [ 13, 8, 3]                    # Number of steps
dts= [ ( tn -t0 )/ n for n in ns]   # Step size

# Numerical solutions for different time steps
u = [ nums.ode.Euler(f,t0,u0,dts[i],ns[i]) for i in range(len(ns)) ]

for it in range(len(ns)):           # Plot
    plt.plot( u[it][0], u[it][1], 'o', label = r'$\Delta t={:3.1f}$'.format(dts[it]) )

plt.legend(loc='best')
plt.title(r"Equation $u'=-u$. Euler $(\Delta t)_\mathrm{max}=2.0$")
plt.xlabel(r"time $t$")
plt.ylabel(r"function $u$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
# system of 1 ode; convergence
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )
plt.title(r"Equation $u'=-u$. Euler $(\Delta t)_\mathrm{max}=2.0$")
plt.plot(te, ue, label='exact')

ns = [ 7, 14, 28 ]                  # Number of steps
dts= [ ( tn -t0 )/ n for n in ns]   # Step size

# Numerical solution
u = [ nums.ode.Euler(f,t0,u0,dts[i],ns[i]) for i in range(len(ns)) ]

for it in range(len(ns)):           # Plot
    plt.plot( u[it][0], u[it][1], 'o', label = r'$\Delta t={:3.2f}$'.format(dts[it]) )

plt.legend(loc='best')
plt.title(r"Equation $u'=-u$. Euler $(\Delta t)_\mathrm{max}=2.0$")
plt.xlabel(r"time $t$")
plt.ylabel(r"function $u$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
#%% system of 2 ode; harmonic oscillator
fig_id = fig_id +1
fig = plt.figure( figsize = figsize11 )
plt.title(r"Equation $u''+u=0$")

def sol(t):
    return np.sin(t)

def f(u,t):     # write 2. order ode as a system of 2 1. order equations
    return np.array([ u[1], -u[0]])

n  = 20                     # Number of steps
dt = ( tn -t0 )/ n          # Step size
u0 = [0., 1.]               # Initial condition

# Exact solution
ue = sol(te) 

# Numerical solution using various schemes
legends =[]                                     # Create list for the schemes
schemes =[]
legends.append( "Euler" )
schemes.append( nums.ode.Euler )
legends.append( "Runge-Kutta 3" )
schemes.append( nums.ode.RungeKutta3 )

u = [ s(f,t0,u0,dt,n) for s in schemes ]        # Integrate 

# Plot functions
plt.plot( te, ue, label=r'exact' )
for i in range(len(schemes)):
    # plt.plot(u[i][0], u[i][1], 'o', label=legends[i] )
   plt.plot(u[i][0], u[i][1][:,0], 'o', label=legends[i] )
plt.legend(loc='best')
plt.xlabel(r"time $t$")
plt.ylabel(r"function $u$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format(tag+str(fig_id)),bbox_inches='tight')

#################################################
plt.show()
