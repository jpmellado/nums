#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Time-marching schemes 

import numpy as np

def Euler(f,t0,u0,dt,itmax):
# Solves an ODE u'=f(u,t) using Euler method
#    - f: python function that calculates f(u,t) 
#    - t0: Initial time
#    - t0: Initial condition
#    - dt: Time step
#    - itmax: Maximum number of iterations
# Output parameters
#    - t: Vector with values of independent variable 
#    - u: Solution vector

    t = np.zeros(itmax+1)
    u = np.zeros((itmax+1,)+np.shape(u0))
    t[0] = t0
    u[0] = u0
    
    for it in range(itmax):
        t[it+1] = t[it] +dt
        u[it+1] = u[it] +dt *f(u[it],t[it])
        
    return t, u

def RungeKutta2(f,t0,u0,dt,itmax):
# Solves an ODE u'=f(u,t) using Runge-Kutta methods
#    - f: python function that calculates f(u,t) 
#    - t0: Initial time
#    - t0: Initial condition
#    - dt: Time step
#    - itmax: Maximum number of iterations
# Output parameters
#    - t: Vector with values of independent variable 
#    - u: Solution vector

    t = np.zeros(itmax+1)
    u = np.zeros((itmax+1,)+np.shape(u0))
    t[0] = t0
    u[0] = u0
    
    for it in range(itmax):
        t[it+1] = t[it] +dt
        k1 = f( u[it],         t[it]  )
        k2 = f( u[it] +dt *k1, t[it+1])
        u[it+1] = u[it] +0.5 *dt *( k1 +k2 )
        
    return t, u

def RungeKutta3(f,t0,u0,dt,itmax):
# Solves an ODE u'=f(u,t) using Runge-Kutta methods
#    - f: python function that calculates f(u,t) 
#    - t0: Initial time
#    - t0: Initial condition
#    - dt: Time step
#    - itmax: Maximum number of iterations
# Output parameters
#    - t: Vector with values of independent variable 
#    - u: Solution vector

    t = np.zeros(itmax+1)
    u = np.zeros((itmax+1,)+np.shape(u0))
    t[0] = t0
    u[0] = u0
    
    for it in range(itmax):
        t[it+1] = t[it] +dt
        k1 = f( u[it],                     t[it]  )
        k2 = f( u[it] +dt *0.5 *k1,        t[it] +0.5 *dt )
        k3 = f( u[it] +dt *( 2. *k2 -k1 ), t[it+1] )
        u[it+1] = u[it] +dt *( k1 +4. *k2 +k3 ) /6.
        
    return t, u

def AdamsBashforth2(f,t0,u0,dt,itmax):
# Solves an ODE u'=f(u,t) using Euler method
#    - f: python function that calculates f(u,t) 
#    - t0: Initial time
#    - t0: Initial condition
#    - dt: Time step
#    - itmax: Maximum number of iterations
# Output parameters
#    - t: Vector with values of independent variable 
#    - u: Solution vector

    t = np.zeros(itmax+1)
    u = np.zeros((itmax+1,)+np.shape(u0))
    t[0] = t0
    u[0] = u0
    
    # Start with an euler
    t[1] = t[0] +dt
    u[1] = u[0] +dt *f(u[0],t[0])
    
    for it in range(1,itmax):
        t[it+1] = t[it] +dt
        u[it+1] = u[it] +0.5 *dt *( 3. *f(u[it],t[it]) -f(u[it-1],t[it-1]) ) 
        
    return t, u
