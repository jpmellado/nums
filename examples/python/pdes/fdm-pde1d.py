#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Examples of Finite Difference Methods for Advection and Diffusion Equations

import numpy as np
import matplotlib.pyplot as plt
from checkpointing import *

# from iodata import *

# Define the temporal grid
tmin = 0.0  # Initial time
tmax = 2.0  # Final time
tcheck = 0.5  # Time interval to checkpoint data

# Define the spatial grid, uniformly spaced
xmin = -1.0
xmax = 1.0
nx = 20
x = np.linspace(xmin, xmax, nx)
h = (x[nx - 1] - x[0]) / (nx - 1)

# Define the problem
diffusivity = None  # Default
velocity = None  # Default
# velocity = 1.0
# from pynums.pdes.advection import *
diffusivity = 0.2
from pynums.pdes.diffusion import *

from pynums.pdes.timemarching import *

# timescheme = RungeKutta3(dnum=0.55)  # , cnum=2.0)
timescheme = Euler(dnum=0.55)  # , cnum=2.0)
dnum = diffusionNumber(timescheme.dt)  # Calculate values with actual delta_t
cnum = advectionNumber(timescheme.dt)


###############################################################################
# Pre-processing


# Initialize loop
t = tmin  # Time
u = ics(x)


it = 0  # To keep track of how many iterations we need

###############################################################################
# Running simulation

# Checkpointing meta-data (save a set of times); using python lists
data = checkPointing(delta_t=tcheck)

data.add(t, u)  # Checkpoint initial condition

# Integration loop
while t < tmax:
    u, t = timescheme.advanceStep(rhs, u, t)  # Advance one time step
    bcs(u, t)  # Corrections to satisfy the boundary conditions, if needed
    it = it + 1

    if t >= data.time2check:  # Checkpoint data
        data.add(t, u)

# Always checkpoint the last time, if not done
if t >= data.time2check:
    data.add(t, u)

###############################################################################
# Post-processing: Plot functions
from postprocessing import *

# Figure names and counter
tag = "fdm-pde1d"
fig_id = 0

fig_id = fig_id + 1
fig, axs = PlotCurves(
    x, data, reference(u, t), "{}.pdf".format(tag + "-" + str(fig_id))
)
