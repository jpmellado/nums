import numpy as np
import scipy.linalg

# Inverts a finite-difference approximation to solve a boundary value problem.

def bvp1_e11b(f):
    # Solves the BVP 
    #       u'_j=f_j,   j = 2,...n 
    #       u_1 = 0
    # using a explicit formulation with backward biased formulas
    # Still need to multiply by the grid spacing h
    # Input arguments:
    #    - f: values of the forcing at the grid points
    # Output arguments:
    #    -u: solution at the grid points 
    n = np.size(f)
    u = np.zeros_like(f)

    # Create system array
    A = np.diag(np.full(n-1, 1.))                       # Create array and fill main diagonal
    np.fill_diagonal(A[1:,:],np.full(n-2,-1.))          # Fill lower diagonal

    # Solve the system. We use generic routines, but one could use solve_banded
    u[1:] = scipy.linalg.solve( A, f[1:] )

    return u
