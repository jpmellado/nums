import numpy as np
import nums.fdm

if __name__ != "__main__":
    from __main__ import x, h, diffusivity

# Diffusion problem with Dirichlet bcs
def ics(x):               # Initial conditions
    f = np.copy(x)
    return f

def bcs(u,t):             # Boundary conditions, nothing to be done
    return

def rhs(u,t):             # Right-hand side of evolution equation (tendency)
    f = np.empty(np.size(x))
    f = 1.                                  # Source term: constant
    f = diffusivity *nums.fdm.fdm2_e121(u) /( h **2. ) +f
    f[ 0] = 0.                              # Calculate boundary conditions; Dirichlet
    f[-1] = 0.                              # Boundary nodes in state vector, set by initial condition, keep them fixed
    return f

def reference(u,t):             # Reference function to compare with: Steady-state solution
    f = -0.5 *(x-1.) *(x+1.) /diffusivity   # Constant source term
    slope = ( u[-1] -u[0] ) /( x[-1] -x[0] )
    f = f +x[0] +slope *( x +1. )           # Linear profile between boundaries
    return f

