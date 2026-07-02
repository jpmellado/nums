#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Finite difference equations. Steady-state solutions of heat equation

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pynums.fdms.bvp2 import *
from pynums.template import *

# Input data
n = 20
u1 = 0.0  # Dirichlet boundary condition at x_1
un = 1.0  # Dirichlet boundary condition at x_n


def s(x):  # Define source term
    s = np.empty(np.size(x))
    s = 10.0 * np.sin(np.pi * x)
    return s


# Create grid
x = np.linspace(0.0, 1.0, n)
h = (x[-1] - x[0]) / (n - 1)

# Create forcing term
b = s(x) * h**2

# Solve the problem
u = bvp2_e121(b, u1, un)

#################################################
# Plotting
fig_id = 0

fig_id = fig_id + 1
fig = plt.figure(figsize=figsize11)

plt.plot(x, u)
plt.xlabel(r"Horizontal distance $x$")
plt.ylabel(r"Temperature $u$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format("heat-" + str(fig_id)), bbox_inches="tight")

#################################################
fig_id = fig_id + 1
fig = plt.figure(figsize=figsize11)

plt.plot(x, s(x), "C1")
plt.xlabel(r"Horizontal distance $x$")
plt.ylabel(r"Heat source $s$")
plt.grid()
plt.tight_layout(pad=0.1)

plt.savefig("{}.pdf".format("heat-" + str(fig_id)), bbox_inches="tight")

plt.show()
