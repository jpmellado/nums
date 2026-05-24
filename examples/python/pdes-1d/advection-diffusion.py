# Advection-diffusion problem, Dirichlet at left boundary and Neumann at the right boundary

import numpy as np
import matplotlib.pyplot as plt
from pynums.pdes.timemarching import *
from pynums.pdes.checkpointing import *
from pynums.pdes.postprocessing import PlotCurves

# from iodata import *

# Define the temporal grid
tmin = 0.0  # Initial time
tmax = 8.0  # Final time
tcheck = 2.0  # Time interval to checkpoint data

# Define the spatial grid, uniformly spaced
xmin = -1.0
xmax = 1.0
nx = 20

# Define the problem
velocity = 1.0
diffusivity = 0.2
# timescheme = Euler(delta_t=0.2)
timescheme = RungeKutta3()
cnum = 0.20  # Choose if you fix diffusion number or time step
dnum = 0.20


###########################################################
def reference(x):
    return 1.0 + np.cos(np.pi * x)


def preprocessing():
    # Construct grid
    x = np.linspace(xmin, xmax, nx)

    # Construct initial condition
    u = reference(x)
    u[:] = 0.0

    return x, u


def simulation(x, u):
    # Create checkpointing object (save a set of times)
    data = checkPointing(delta_t=tcheck)

    # Calculate time step, if needed
    if timescheme.dt == None:
        timescheme.dt = timeStep(
            x, velocity=velocity, cnum=cnum, diffusivity=diffusivity, dnum=dnum
        )

    # Integration loop
    t = tmin
    it = 0  # To keep track of how many iterations we need
    data.add(it, t, u)  # Checkpoint initial condition

    while t < tmax:
        u, t = timescheme.advanceStep(rhs, u, t)  # Advance one time step
        bcs(u, t)  # Corrections to satisfy the time-varying boundary conditions, if needed
        it = it + 1

        if t >= data.time2check:  # Checkpoint data
            data.add(it, t, u)
            data.time2check = min(data.time2check, tmax)  # To always checkpoint last time

    return data


def postprocessing(x, data):
    fig, axs = PlotCurves(x, data)

    cnum = advectionNumber(x, velocity, timescheme.dt)  # Calculate values with actual delta_t
    axs.set_title(
        r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(
            cnum, np.size(x), data.ichecked[-1]
        )
    )

    plt.tight_layout(pad=0.1)
    plt.savefig("advection.pdf", bbox_inches="tight")
    plt.show()


###########################################################
h = (xmax - xmin) / (nx - 1)  # define global variable used below in rhs


def source(x, t):
    s = np.empty_like(x)
    s[:] = 0.0  # no source
    return s


def bcs(u, t):
    u[0] = reference(x[0] - velocity * t)  # left, time-varying Dirichlet
    u[-1] = (4.0 * u[-2] - u[-3]) / 3.0 + 2.0 / 3.0 * h / diffusivity  # right, Neumann

    return


def rhs(u, t):  # Right-hand side of evolution equation (tendency)
    from pynums.fdms.fdm1 import fdm1_e121
    from pynums.fdms.fdm2 import fdm2_e121

    # boundary conditions
    bcs(u, t)  # necessary for substages in RK

    # Using second-order approximations to spatial derivatives
    # Maximum cfl number is imagMax
    f = -velocity * fdm1_e121(u) / h + diffusivity * fdm2_e121(u) / (h**2.0) + source(x, t)

    # boundary conditions
    f[0] = 0.0  # Dirichlet, set by function bcs
    f[-1] = 0.0  # this cannot be right

    return f


###########################################################
if __name__ == "__main__":
    x, u = preprocessing()
    data = simulation(x, u)
    postprocessing(x, data)
