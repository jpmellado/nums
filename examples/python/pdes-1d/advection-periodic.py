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
nx = 19

# Define the problem
velocity = 1.0
# timescheme = Euler(delta_t=0.2)
timescheme = RungeKutta3()
cnum = 0.20  # Choose if you fix diffusion number or time step


###########################################################
def reference(x):
    return np.cos(np.pi * x)


def preprocessing():
    # Construct grid (periodic)
    x = np.linspace(xmin, xmax, nx + 1)[:-1]
    print(np.size(x), x[-1])

    # Construct initial condition
    u = reference(x)

    return x, u


def simulation(x, u):
    # Create checkpointing object (save a set of times)
    data = checkPointing(delta_t=tcheck)

    # Calculate time step, if needed
    if timescheme.dt == None:
        timescheme.dt = timeStep(x, velocity=velocity, cnum=cnum)

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

    cnum = advectionNumber(x, velocity, timescheme.dt)  # Calculate values with actual delta_t
    axs.set_title(
        r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(
            cnum, np.size(x), data.ichecked[-1]
        )
    )

    # overwrite default to handle periodic conditions
    axs.set_xlim(xmin, xmax)
    axs.set_xticks(np.linspace(xmin, xmax, num=5))

    # Add exact solution as reference
    f = reference(x - velocity * data.tchecked[-1])

    ref = axs.plot(x, f, color="black", label=r"reference")
    axs.legend()

    plt.tight_layout(pad=0.1)
    plt.savefig("advection.pdf", bbox_inches="tight")
    plt.show()


###########################################################
h = (xmax - xmin) / nx  # define global variable used below in rhs


def source(x, t):  # periodic source
    s = np.empty_like(x)
    s[:] = 0.0  # no source
    return s


def rhs(u, t):  # Right-hand side of evolution equation (tendency)
    from pynums.fdms.fdm1 import fdm1_e2p

    f = np.empty_like(u)

    # Using second-order approximations to spatial derivatives
    # Maximum diffusion number is realMax/4
    f = -velocity * fdm1_e2p(u) / h + source(x, t)

    return f


###########################################################
if __name__ == "__main__":
    x, u = preprocessing()
    data = simulation(x, u)
    postprocessing(x, data)
