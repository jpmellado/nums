import numpy as np
import nums.fdm

if __name__ != "__main__":
    from __main__ import x, h, velocity

# Advection problem with periodic bcs

def ics(x):                     # Initial conditions
    f = np.empty(np.size(x))
    f = np.cos( np.pi *x )
    return f

def bcs(u,t):                   # Boundary conditions, nothing to be done
    return

def rhs(u,t):                   # Right-hand side of evolution equation (tendency)
    n = np.size(x)
    f = np.empty(n)    
    f[:n-1] = -velocity *nums.fdm.fdm1_e2p(u[:n-1]) /h
    f[n-1] = f[0]               # Calculate boundary conditions; periodic
    return f

def reference(u,t):             # Reference function to compare with: Exact solution for constant advection
    f = ics(x-velocity *t)
    return f

