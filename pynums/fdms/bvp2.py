import numpy as np
import scipy.linalg

# Inverts a finite-difference approximation to 2. order derivative to solve a boundary value problem.


def bvp2_e121(f, alpha, beta):
    # Solves the BVP
    #       u''_j=f_j,   j = 2,...n-1
    #       u_1 = alpha
    #       u_n = beta
    # using a explicit formulation with centered formulas
    # Still need to multiply the source term by the grid spacing h^2
    # Input arguments:
    #    - f: values of the forcing at the grid points
    # Output arguments:
    #    -u: solution at the grid points
    n = np.size(f)
    u = np.zeros_like(f)

    # Create system array
    A = np.diag(np.full(n - 2, 2.0))  # Create array and fill main diagonal
    np.fill_diagonal(A[1:, :], np.full(n - 3, -1.0))  # Fill lower diagonal
    np.fill_diagonal(A[:, 1:], np.full(n - 3, -1.0))  # Fill upper diagonal

    # add boundary conditions to forcing
    f1 = np.copy(f)
    f1[1] = f[1] + alpha
    f1[-2] = f[-2] + beta

    # Solve the system. We use generic routines, but one could use solve_banded
    u[1:-1] = scipy.linalg.solve(A, f1[1:-1], assume_a="banded")

    # impose boundary conditions
    u[0] = alpha
    u[-1] = beta

    return u


def bvp2_e121_dn(f, alpha, beta):  # dirichlet-neumann conditions
    # Solves the BVP
    #       u''_j=f_j,   j = 2,...n-1
    #       u_1 = alpha
    #       u'_n = beta
    # using a explicit formulation with centered formulas
    # Still need to multiply the source term by the grid spacing h^2
    # and the neumann condition by h
    # Input arguments:
    #    - f: values of the forcing at the grid points
    # Output arguments:
    #    -u: solution at the grid points
    n = np.size(f)
    u = np.zeros_like(f)

    # Create system array
    A = np.diag(np.full(n - 1, 2.0))  # Create array and fill main diagonal
    np.fill_diagonal(A[1:, :], np.full(n - 2, -1.0))  # Fill lower diagonal
    np.fill_diagonal(A[:, 1:], np.full(n - 2, -1.0))  # Fill upper diagonal
    A[-1, -1] = 1.0  # correction to impose neumman condition

    # add boundary conditions to forcing
    f1 = np.copy(f)
    f1[1] = f[1] + alpha
    f1[-1] = beta

    # Solve the system. We use generic routines, but one could use solve_banded
    u[1:] = scipy.linalg.solve(A, f1[1:], assume_a="banded")

    # impose boundary conditions
    u[0] = alpha

    return u
