# Advection with constant velocity vector in a double periodic domain

import numpy as np
import matplotlib.pyplot as plt
from pynums.pdes.timemarching import *
from pynums.pdes.checkpointing import *
from pynums.pdes.postprocessing import *

# from iodata import *

# Define the temporal grid
tmin = 0.0  # Initial time
tmax = 2.0  # Final time
tcheck = 0.5  # Time interval to checkpoint data

# Define the spatial grid, uniformly spaced
xmin = -1.0
xmax = 1.0
nx = 19
ymin = -1.0
ymax = 1.0
ny = 29

# Define the problem
velocity = 1.0
angle = np.pi / 4.0
velocity_x = velocity * np.cos(angle)
velocity_y = velocity * np.sin(angle)
# timescheme = Euler(delta_t=0.2)
timescheme = RungeKutta3()
cnum = 0.20  # Choose if you fix diffusion number or time step


###########################################################
def reference(grid):
    xc = grid[0]  # for clarity
    yc = grid[1]
    return np.cos(np.pi * xc) * np.cos(2.0 * np.pi * yc)


def preprocessing():
    # Construct grid (periodic)
    x = np.linspace(xmin, xmax, nx + 1)[:-1]
    y = np.linspace(ymin, ymax, ny + 1)[:-1]
    grid = np.meshgrid(x, y)  # 2-tuple containing xc, yc

    # Construct initial condition
    u = reference(grid)

    return grid, u


def simulation(grid, u):
    xc = grid[0]  # for clarity
    yc = grid[1]

    # Create checkpointing object (save a set of times)
    data = checkPointing(delta_t=tcheck)

    # Calculate time step, if needed
    if timescheme.dt == None:
        dt_x = timeStep(xc, velocity=velocity_x, cnum=cnum)
        dt_y = timeStep(yc.T, velocity=velocity_y, cnum=cnum)
        timescheme.dt = min(dt_x, dt_y)

    # Integration loop
    t = tmin
    it = 0  # To keep track of how many iterations we need
    data.add(it, t, u)  # Checkpoint initial condition

    while t < tmax:
        u, t = timescheme.advanceStep(rhs, u, t)  # Advance one time step
        it = it + 1

        if t >= data.time2check:  # Checkpoint data
            data.add(it, t, u)
            data.time2check = min(data.time2check, tmax)  # To always checkpoint last time

    return data


def postprocessing(grid, data):
    xc = grid[0]  # for clarity
    yc = grid[1]

    fig, axs = PlotContours(grid, data)

    cnum_x = advectionNumber(xc, velocity, timescheme.dt)  # Calculate values with actual delta_t
    cnum_y = advectionNumber(yc.T, velocity, timescheme.dt)  # Calculate values with actual delta_t
    cnum = max(cnum_x, cnum_y)

    axs.set_title(
        r"CFL \# ${:3.2f}$. ${:d}\times{:d}$ grid points. ${:d}$ iterations.".format(
            cnum, nx, ny, data.ichecked[-1]
        )
    )

    # overwrite default to handle periodic conditions
    axs.set_xlim(xmin, xmax)
    axs.set_xticks(np.linspace(xmin, xmax, num=5))
    axs.set_ylim(ymin, ymax)
    axs.set_yticks(np.linspace(ymin, ymax, num=5))

    plt.tight_layout(pad=0.1)
    plt.savefig("advection.pdf", bbox_inches="tight")
    plt.show()


###########################################################
from pynums.fdms.fdm1 import fdm1_e2p
from pynums.partials import partial

hx = (xmax - xmin) / nx  # define global variable used below in rhs
hy = (ymax - ymin) / ny

# Using second-order approximations to spatial derivatives
# Maximum diffusion number is realMax/4
partial_1 = partial(scheme=fdm1_e2p, step_x=hx, step_y=hy)


def source(grid, t):  # periodic source
    xc = grid[0]  # for clarity
    yc = grid[1]
    s = np.empty_like(xc)
    s[:, :] = 0.0  # no source
    return s


def rhs(u, t):  # Right-hand side of evolution equation (tendency)
    f = -velocity_x * partial_1.dx(u) - velocity_y * partial_1.dy(u) + source(grid, t)

    return f


###########################################################
if __name__ == "__main__":
    grid, u = preprocessing()
    data = simulation(grid, u)
    postprocessing(grid, data)
