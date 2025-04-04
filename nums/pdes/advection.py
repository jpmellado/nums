import numpy as np
import nums.fdm

if __name__ != "__main__":
    from __main__ import x, h, velocity

# Advection problem with non-periodic bcs

def ics(x):                        # Initial conditions
    f = np.empty(np.size(x))
    f[:] = 0.
    return f

def bcs(u,t):                      # Boundary conditions, time-varying Dirichlet
    u[0] = 1. -np.cos( np.pi *t )
    return

def rhs(u,t):                      # Right-hand side of evolution equation (tendency)
    n = np.size(x)
    f = np.empty(n)    
    bcs(u,t)                       # Ensure bcs; necessary for substages in RK
    f = -velocity *nums.fdm.fdm1_e121(u) /h 
    f[0] = 0.                      # Calculate boundary conditions; Dirichlet at left boundary
    return f

def reference(u,t):                # Reference function to compare with: Exact solution for constant advection
    f = np.empty(np.size(x))
    f = 1. -np.cos( np.pi *( t -(x -x[0])/velocity ) )
    f = np.where( x < x[0] +velocity *t, f, 0. )
    return f

