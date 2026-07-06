import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pynums.fdms.bvp2 import *
from pynums.template import *

# Define the spatial grid, uniformly spaced
xmin = 0.0
xmax = 2.0
nx = 30
ymin = 0.0
ymax = 1.0
ny = 10

print("Not working yet...")


# Define the problem
# source term
def s(x, y):  # trigonometric function for x-periodic conditions
    x = grid[0]  # for clarity
    y = grid[1]

    s = np.empty_like(x)
    s = np.sin(np.pi * x / xmax) ** 2  # * np.sin(np.pi * y / ymax) ** 2

    return s


# neumann boundary conditions
def bcs0(x):
    return np.zeros_like(x)


def bcs1(x):
    return np.ones_like(x)


###########################################################
def preprocessing():
    # Create grid
    x1 = np.linspace(xmin, xmax, nx + 1)[:-1]
    y1 = np.linspace(ymin, ymax, ny)
    grid = np.meshgrid(x1, y1)  # 2-tuple containing xc, yc

    return grid


###########################################################
hx = (xmax - xmin) / nx  # define global variable used below in rhs
hy = (ymax - ymin) / (ny - 1)


def solve(grid, s):
    x = grid[0]  # for clarity
    y = grid[1]

    source = s(x, y)  # source term
    fs = np.fft.rfft(source)

    u1 = bcs0(x[0, :])  # boundary condition at the bottom
    fu1 = np.fft.rfft(u1)

    un = bcs0(x[0, :])  # boundary conditions at the top
    fun = np.fft.rfft(un)

    # solve fft x-transform system for each mode
    fu = np.empty_like(fs)

    w0 = 2.0j * np.pi / nx / hx

    ki = 0
    fu[:, ki] = bvp2_e121_dn(fs[:, ki] * hy**2, 0.0, hy * fun[ki])

    for ki in range(1, np.size(fs, 1)):
        fu[:, ki] = bvp2extended_e121_nn(
            fs[:, ki] * hy**2, (ki * w0) ** 2 * hy**2, hy * fu1[ki], hy * fun[ki]
        )

    u = np.fft.irfft(fu)

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
