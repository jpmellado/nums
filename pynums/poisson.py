import numpy as np
import scipy.linalg
from pynums.fdms.bvp2 import *


def poisson_periodic(grid, s, bcs):
    # Solves Poisson equation lap u = s in two-dimensional domain with
    # periodic boundary conditions in x and
    # Neumann boundary conditions in y
    # Make sure compatibility constraint int (bcs_top -bcs_bottom) = int s
    #   - grid : 2-tuple with coordinates of grid points
    #   - s: function to calculate the right-hand side
    #   - bcs: 2-tuple with functions to calculate boundary conditions
    # Output
    #   - u: solution of the Poisson equation

    x = grid[0]  # for clarity
    y = grid[1]
    bcs_b = bcs[0]  # function defining bottom boundary conditions
    bcs_t = bcs[1]  # function defining top boundary conditions

    nx = np.size(x[0, :])
    hx = x[0, 1] - x[0, 0]  # assuming uniform grid
    hy = y[1, 0] - y[0, 0]

    source = s(x, y)  # source term
    fs = np.fft.rfft(source)

    u1 = bcs_b(x[0, :])  # boundary condition at the bottom
    fu1 = np.fft.rfft(u1)

    un = bcs_t(x[0, :])  # boundary conditions at the top
    fun = np.fft.rfft(un)

    # solve fft x-transform system for each mode
    fu = np.empty_like(fs)

    w0 = 2.0 * np.pi / nx / hx

    ki = 0
    fu[:, ki] = bvp2_e121_dn(fs[:, ki] * hy**2, 0.0, hy * fun[ki])

    for ki in range(1, np.size(fs, 1)):
        fu[:, ki] = bvp2extended_e121_nn(
            fs[:, ki] * hy**2, (ki * w0) ** 2 * hy**2, hy * fu1[ki], hy * fun[ki]
        )

    u = np.fft.irfft(fu)

    return u


def poisson_dd(grid, s, bcs):
    # Solves Poisson equation lap u = s in two-dimensional domain with
    # Dirichlet boundary conditions in x and y
    #   - grid : 2-tuple with coordinates of grid points
    #   - s: function to calculate the right-hand side
    #   - bcs: 4-tuple with functions to calculate boundary conditions
    # Output
    #   - u: solution of the Poisson equation

    x = grid[0]  # for clarity
    y = grid[1]
    bcs_b = bcs[0]  # function defining bottom boundary conditions
    bcs_t = bcs[1]  # function defining top boundary conditions
    bcs_l = bcs[2]  # function defining left (west) boundary conditions
    bcs_r = bcs[3]  # function defining right (east) boundary conditions

    nx = np.size(x[0, :])
    ny = np.size(y[:, 0])
    hx = x[0, 1] - x[0, 0]  # assuming uniform grid
    hy = y[1, 0] - y[0, 0]

    # Create array for the field
    u = np.empty_like(x)
    u[:, 0] = bcs_l(y[:, 0])  # Dirichlet boundary condition at x_1
    u[:, -1] = bcs_r(y[:, 0])  # Dirichlet boundary condition at x_n
    u[0, :] = bcs_b(x[0, :])  # Dirichlet boundary condition at y_1
    u[-1, :] = bcs_t(x[0, :])  # Dirichlet boundary condition at y_n

    # Create system array
    n = (nx - 2) * (ny - 2)
    Ax = np.diag(np.full(n, -2.0))  # Create array and fill main diagonal
    np.fill_diagonal(Ax[1:, :], np.full(n - 1, 1.0))  # Fill lower diagonal
    Ax[nx - 2 :: nx - 2, nx - 3 :: nx - 2] = 0.0
    np.fill_diagonal(Ax[:, 1:], np.full(n - 1, 1.0))  # Fill upper diagonal
    Ax[nx - 3 :: nx - 2, nx - 2 :: nx - 2] = 0.0
    # print(Ax)

    Ay = np.diag(np.full(n, -2.0))  # Create array and fill main diagonal
    np.fill_diagonal(Ay[nx - 2 :, :], np.full(n - (nx - 2), 1.0))  # Fill lower diagonal
    np.fill_diagonal(Ay[:, nx - 2 :], np.full(n - (nx - 2), 1.0))  # Fill upper diagonal
    # print(Ay)

    A = (1.0 / hx**2) * Ax + (1.0 / hy**2) * Ay

    # Create array for the source
    source = s(x, y)

    # Create forcing term
    b = source[1:-1, 1:-1].flatten()

    b[0 :: nx - 2] -= (1.0 / hx**2) * u[1 : ny - 1, 0]  # Dirichlet boundary condition at x_1
    b[nx - 3 :: nx - 2] -= (1.0 / hx**2) * u[1 : ny - 1, -1]  # Dirichlet boundary condition at x_n
    b[: nx - 2] -= (1.0 / hy**2) * u[0, 1 : nx - 1]  # Dirichlet boundary condition at y_1
    b[n - (nx - 2) :] -= (1.0 / hy**2) * u[-1, 1 : nx - 1]  # Dirichlet boundary condition at y_n
    # print(b)

    # Solve the system. We use generic routines, but one could use solve_banded
    sol = scipy.linalg.solve(A, b)
    u[1:-1, 1:-1] = sol.reshape((ny - 2, nx - 2))

    return u
