import numpy as np
import nums.fdm

if __name__ != "__main__":
    from __main__ import x, h, velocity, diffusivity

# Advection-diffusion problem, Dirichlet at left boundary and Neumann at the right boundary
def ics(x):               # Initial conditions
    f = np.empty(np.size(x))
    f = (x -x[0]) /diffusivity
    return f

def bcs(u,t):             # Boundary conditions, Neumann
    u[-1] = ( 4.0 * u[-2] - u[-3] )/ 3. + 2./3. *h /diffusivity
    return

def rhs(u,t):             # Right-hand side of evolution equation (tendency)
    n = np.size(x)
    f = np.empty(n)    
    bcs(u,t)                                # Ensure bcs; necessary for substages in RK
    f = -velocity *(1-x) *nums.fdm.fdm1_e121(u) /h +diffusivity *nums.fdm.fdm2_e121(u) /( h **2. ) 
    f[ 0] = 0.                              # Bcs: no change imposed here             
    f[-1] = 0.
    return f

def reference(u,t):             # Reference function to compare with: Exact solution for constant advection
    f = np.empty(np.size(x))
    f[:] = 0.
    return f

