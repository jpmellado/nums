#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite difference equations.
# Steady-state solutions of heat equation lap u + s = 0

import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from pynums.poisson import *
from pynums.template import *

# Define the spatial grid, uniformly spaced
xmin = 0.0
xmax = 2.0
nx = 30
ymin = 0.0
ymax = 1.0
ny = 10


# Define the problem
# source term
def s(x, y):  # trigonometric function for x-periodic conditions
    x = grid[0]  # for clarity
    y = grid[1]

    s = np.empty_like(x)
    s = np.sin(2.0 * np.pi * x / xmax) * np.sin(np.pi * y / ymax) ** 2
    # s = np.sin(2.0 * np.pi * x / xmax) * (2.0 * np.pi / xmax) ** 2

    return s


def s_minus(x, y):
    return -s(x, y)


# neumann boundary conditions
def bcs0(x):
    return np.zeros_like(x)


def bcs1(x):
    return np.ones_like(x)


bcs = (bcs0, bcs0)


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
    plt.savefig("{}.pdf".format("heat-" + str(fig_id)), bbox_inches="tight")

    #################################################
    fig_id = fig_id + 1
    fig = plt.figure(figsize=figsize11)
    plt.title(r"source $s$")

    c = plt.contourf(x, y, s(x, y))
    plt.colorbar(c, label=r"$s$")
    formatPlot()

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format("heat-" + str(fig_id)), bbox_inches="tight")

    plt.show()


def formatPlot():
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()


###########################################################
if __name__ == "__main__":
    grid = preprocessing()
    #
    x = grid[0]  # for clarity
    y = grid[1]
    print("Compatibility constraint (both below should be equal).")
    print("Source integrates to {:f} ".format(simpson(np.sum(s_minus(x, y), axis=1), y[:, 0])))
    print(
        "Difference in boundary conditions is {:f}".format(
            np.sum(bcs[1](x[0, :]) - bcs[0](x[0, :]))
        )
    )
    #
    u = poisson_periodic(grid, s_minus, bcs)
    postprocessing(grid, u, s)
