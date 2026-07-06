#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite difference equations. Steady-state solutions of heat equation

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pynums.fdms.bvp2 import *
from pynums.template import *

# Define the spatial grid, uniformly spaced
xmin = 0.0
xmax = 2.0
nx = 20
ymin = 0.0
ymax = 1.0
ny = 10


# Define the problem: source term
def s(x, y):  # trigonometric function for dirichlet conditions
    x = grid[0]  # for clarity
    y = grid[1]

    s = np.empty_like(x)
    s = np.sin(np.pi * x / xmax) ** 2 * np.sin(np.pi * y / ymax) ** 2
    return s


###########################################################
def preprocessing():
    # Create grid
    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)
    grid = np.meshgrid(x1, y1)

    return grid


###########################################################
hx = (xmax - xmin) / (nx - 1)  # define global variable used below in rhs
hy = (ymax - ymin) / (ny - 1)


def solve(grid, s):
    x = grid[0]  # for clarity
    y = grid[1]

    # Create array for the field
    u = np.empty_like(x)
    u[:, 0] = 1.0  # Dirichlet boundary condition at x_1
    u[:, -1] = 1.0  # Dirichlet boundary condition at x_n
    u[0, :] = 1.0  # Dirichlet boundary condition at y_1
    u[-1, :] = 1.0  # Dirichlet boundary condition at y_n

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
    b = -source[1:-1, 1:-1].flatten()

    b[0 :: nx - 2] -= (1.0 / hx**2) * u[
        1 : ny - 1, 0
    ]  # Dirichlet boundary condition at x_1
    b[nx - 3 :: nx - 2] -= (1.0 / hx**2) * u[
        1 : ny - 1, -1
    ]  # Dirichlet boundary condition at x_n
    b[: nx - 2] -= (1.0 / hy**2) * u[
        0, 1 : nx - 1
    ]  # Dirichlet boundary condition at y_1
    b[n - (nx - 2) :] -= (1.0 / hy**2) * u[
        -1, 1 : nx - 1
    ]  # Dirichlet boundary condition at y_n
    # print(b)

    # Solve the system. We use generic routines, but one could use solve_banded
    sol = scipy.linalg.solve(A, b)
    u[1:-1, 1:-1] = sol.reshape((ny - 2, nx - 2))

    return u


def postprocessing(grid, u, s):
    x = grid[0]  # for clarity
    y = grid[1]

    fig_id = 0

    fig_id = fig_id + 1
    fig = plt.figure(figsize=figsize11)
    plt.title(r"solution of $\nabla^2u +s=0$")

    c = plt.contourf(x, y, u)
    plt.colorbar(c, label=r"$u$")
    formatPlot()

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format("poisson-" + str(fig_id)), bbox_inches="tight")

    #################################################
    fig_id = fig_id + 1
    fig = plt.figure(figsize=figsize11)
    plt.title(r"source $s$")

    c = plt.contourf(x, y, s(x, y))
    plt.colorbar(c, label=r"$s$")
    formatPlot()

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format("poisson-" + str(fig_id)), bbox_inches="tight")

    plt.show()

def formatPlot():
    plt.xlabel(r"Horizontal distance $x$")
    plt.ylabel(r"Horizontal distance $y$")
    plt.grid()

###########################################################
if __name__ == "__main__":
    grid = preprocessing()
    u = solve(grid, s)
    postprocessing(grid, u, s)
