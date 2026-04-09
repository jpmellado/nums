#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:29:21 2019

@author: jpmellado
"""

import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt

def SteepestDescent(f,df,x0,itmax):
# Input parameters
#    - f: python function that calculates f(x) 
#    - df: python function that calculates df(x)
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration
    
    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    def df_1d(alpha,x,s):
        return f( x +alpha *s )
    
    # Initialize
    p = -df(x)                                # No need to distinguish between gradient r and descent direction
    e[0] = scipy.linalg.norm(p)               # error estimate for initial guess

    # Obtain the remaining approximations
    for it in range(itmax):
        alpha = scipy.optimize.minimize_scalar(df_1d,args=(x,p),bounds=(0.,1.0),method='bounded')
        x = x +alpha.x *p                     # update position
        plt.plot([x[0]-alpha.x*p[0],x[0]],[x[1]-alpha.x*p[1],x[1]],'C0')
        p =-df(x)                             # update residual (gradient direction)

        e[it+1] = scipy.linalg.norm(p)        # error estimate in column space        
    
    return x, e
    
def FletcherReeves(f,df,x0,itmax):
# Input parameters
#    - f: python function that calculates f(x) 
#    - df: python function that calculates df(x)
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration

    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    def df_1d(alpha,x,s):
        return f( x +alpha *s )
    
    # Initialize
    r = df(x)
    p =-np.copy(r)
    e[0] = np.dot(r,r)                        # error estimate is part of the algorithm
    
    # Obtain the remaining approximations    
    for it in range(itmax):
        alpha = scipy.optimize.minimize_scalar(df_1d,args=(x,p),bounds=(0.,1.0),method='bounded')
        x = x +alpha.x *p                     # update position
        plt.plot([x[0]-alpha.x*p[0],x[0]],[x[1]-alpha.x*p[1],x[1]],'C1')
        r =-df(x)                             # update residual (gradient direction)
        e[it+1] = np.dot(r,r)
        if e[it+1] > 1e-16:
            beta = e[it+1] / e[it]
            p = r + beta *p                   # obtain new conjugate direction
        else:
            break
        
    e = np.sqrt(e)                            # error estimate in column space
            
    return x, e

def Newton(f,df,d2f,x0,itmax):
# Input parameters
#    - f: python function that calculates f(x) 
#    - df: python function that calculates df(x)
#    - d2f: python function that calculates the Hessian of f(x)
#    - x0: Initial condition
#    - itmax: Maximum number of iterations
# Output parameters
#    - x: Solution vector
#    - e: Norm of residual at each iteration
    
    x = np.copy(x0)
    e = np.zeros(itmax+1)
    
    def df_1d(alpha,x,s):
        return f( x +alpha *s )
    
    # Initialize
    r = df(x)
    e[0] = scipy.linalg.norm(r)               # error estimate for initial guess

    # Obtain the remaining approximations
    for it in range(itmax):
        p =-scipy.linalg.solve(d2f(x),r)      # update search direction
        alpha = scipy.optimize.minimize_scalar(df_1d,args=(x,p),bounds=(0.,1.0),method='bounded')
        x = x +alpha.x *p                     # update position
        plt.plot([x[0]-alpha.x*p[0],x[0]],[x[1]-alpha.x*p[1],x[1]],'C2')
        r = df(x)                             # update residual (gradient direction)

        e[it+1] = scipy.linalg.norm(r)        # error estimate in column space        
    
    return x, e
