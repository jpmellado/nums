#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:44:09 2019

@author: jpmellado
"""

import numpy as np
import scipy.linalg

def Bisection(f,x0,x1,itmax):
# Solves a non-linear equation using bisection method
#    - f: python function that calculates f(x) 
#    - x0: First guess
#    - x1: Second guess, such that function changes sign
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration
    
    if f(x0)*f(x1) > 0:
        print("Error: The interval may not enclose any root.")
        return
    
    e = np.zeros(itmax+2)
    e[0] = np.abs(f(x0))
    e[1] = np.abs(f(x1))
    
    for it in range(itmax):
        x = 0.5 *( x0 + x1 )
        if f(x0)*f(x) < 0:
            x1 = x
        else:
            x0 = x
            
        e[it+2] = np.abs(f(x))
        
    return x, e

def SuccessiveApproximations(f,x0,itmax):
# Solves a non-linear equation using Newton-Raphson method
#    - f: python function that calculates f(x) 
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    x = x0
    e = np.zeros(itmax+1)
    e[0] = f(x0)
    
    for it in range(itmax):
        x = x - e[it]
        
        e[it+1] = f(x)
        
    e = np.abs(e)
    
    return x, e

def NewtonRaphson1D(f,df,x0,itmax):
# Solves a non-linear equation using Newton-Raphson method
#    - f: python function that calculates f(x) 
#    - df: python function that calculates df(x)
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    x = x0
    e = np.zeros(itmax+1)
    e[0] = f(x0)
    
    for it in range(itmax):
        x = x - e[it] /df(x)
        
        e[it+1] = f(x)
        
    e = np.abs(e)
    
    return x, e

def NewtonRaphson(f,df,x0,itmax):
# Solves a non-linear equation using Newton-Raphson method
#    - f: python function that calculates f(x) 
#    - df: python function that calculates Jacobian df(x)
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    fun = f(x)
    e[0] = scipy.linalg.norm(fun)
    
    for it in range(itmax):
        jac = df(x)
        x = x - scipy.linalg.solve(jac,fun)
        fun = f(x)
        
        e[it+1] = scipy.linalg.norm(fun)
    
    return x, e

def Secant1D(f,x0,x1,itmax):
# Solves a non-linear equation using secant method
#    - f: python function that calculates f(x) 
#    - x0: First guess
#    - x1: Second guess
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    x = [ x0, x1 ]
    e = np.zeros(itmax+2)
    e[0] = f(x0)
    e[1] = f(x1)
    
    for it in range(itmax):
        df = ( e[it+1] -e[it] ) /( x[1] - x[0] )
        x[0] = x[1]
        x[1] = x[1] - e[it+1] /df
        
        e[it+2] = f(x[1])
        
    e = np.abs(e)
    
    return x[1], e
