import numpy as np
import matplotlib.pyplot as plt
from pynums.template import *

# Define the spatial grid, uniformly spaced
xmin = 0.0
xmax = 2.0
nx = 10
ymin = 0.0
ymax = 1.0
ny = 5


# Define the problem: source term
k0 = 3.0  # wavenumber


def s(x, y):  # trigonometric function for x-periodic conditions
    x = grid[0]  # for clarity
    y = grid[1]

    s = np.empty_like(x)
    s = np.sin(2.0 * np.pi * k0 * x / xmax)
    return s


def s1(x, y):
    x = grid[0]
    y = grid[1]

    s = np.empty_like(x)
    s = np.cos(2.0 * np.pi * k0 * x / xmax) * 2.0 * np.pi * k0 / xmax
    return s


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

    # Create array for the source
    source = s(x, y)

    fu = np.fft.rfft(source)
    w0 = 2.0j * np.pi / nx / hx  # fundamental scaled wavenumber
    for j in range(ny):
        for ki in range(np.size(fu, 1)):
            fu[j, ki] = fu[j, ki] * ki * w0

    u = np.fft.irfft(fu)

    return u


def postprocessing(grid, u, s):
    x = grid[0]  # for clarity
    y = grid[1]

    fig_id = 0

    fig_id = fig_id + 1
    fig = plt.figure(figsize=figsize11)

    c = plt.contourf(x, y, u)
    plt.xlabel(r"Horizontal distance $x$")
    plt.ylabel(r"Horizontal distance $y$")
    plt.title(r"solution of $\nabla^2u +s=0$")
    plt.grid()
    plt.colorbar(c, label=r"$u$")

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format("poisson-" + str(fig_id)), bbox_inches="tight")

    #################################################
    fig_id = fig_id + 1
    fig = plt.figure(figsize=figsize11)

    c = plt.contourf(x, y, s1(x, y))
    plt.xlabel(r"Horizontal distance $x$")
    plt.ylabel(r"Horizontal distance $y$")
    plt.title(r"source $s$")
    plt.grid()
    plt.colorbar(c, label=r"$s$")

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format("poisson-" + str(fig_id)), bbox_inches="tight")

    plt.show()


###########################################################
if __name__ == "__main__":
    grid = preprocessing()
    u = solve(grid, s)
    postprocessing(grid, u, s)
