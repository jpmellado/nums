# Diffusion problem, constant Dirichlet at both boundaries

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
diffusivity = 0.2
# timescheme = Euler(delta_t=0.2)
timescheme = Euler()
dnum = 0.20  # Choose if you fix diffusion number or time step


###########################################################
def preprocessing():
    # Construct grid
    x = np.linspace(xmin, xmax, nx)

    # Construct initial condition
    u = x

    return x, u


def simulation(x, u):
    # Create checkpointing object (save a set of times)
    data = checkPointing(delta_t=tcheck)

    # Calculate time step, if needed
    if timescheme.dt == None:
        timescheme.dt = timeStep(x, diffusivity=diffusivity, dnum=dnum)

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


def postprocessing(x, data):
    fig, axs = PlotCurves(x, data)

    dnum = diffusionNumber(x, diffusivity, timescheme.dt)  # Calculate values with actual delta_t
    axs.set_title(
        r"Diffusion \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(
            dnum, np.size(x), data.ichecked[-1]
        )
    )

    # Add exact solution as reference; steady-state solution
    f = -0.5 * (x - 1.0) * (x + 1.0) / diffusivity  # Constant source term
    slope = (u[-1] - u[0]) / (x[-1] - x[0])
    f = f + x[0] + slope * (x + 1.0)  # Linear profile between boundaries

    ref = axs.plot(x, f, color="black", label=r"reference")
    axs.legend()

    plt.tight_layout(pad=0.1)
    plt.savefig("diffusion.pdf", bbox_inches="tight")
    plt.show()


###########################################################
h = (xmax - xmin) / (nx - 1)  # define global variable used below in rhs


def source(x, t):
    s = np.empty_like(x)
    s[:] = 1.0  # homogeneous source
    return s


def rhs(u, t):  # Right-hand side of evolution equation (tendency)
    from pynums.fdms.fdm2 import fdm2_e121

    # Using second-order approximations to spatial derivatives
    # Maximum diffusion number is realMax/4
    f = diffusivity * fdm2_e121(u) / (h**2.0) + source(x, t)

    # boundary conditions
    f[0] = 0.0  # Dirichlet, set by initial conditions
    f[-1] = 0.0  # Dirichlet, set by initial conditions

    return f


###########################################################
if __name__ == "__main__":
    x, u = preprocessing()
    data = simulation(x, u)
    postprocessing(x, data)
