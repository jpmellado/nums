import numpy as np
import matplotlib.pyplot as plt
from pynums.pdes.timemarching import *
from pynums.pdes.checkpointing import *
from pynums.pdes.postprocessing import *
from pynums.iodata import *

# Define the temporal grid
tmin = 0.0  # Initial time
tmax = 2.0  # Final time
tcheck = 0.05  # Time interval to checkpoint data

# Define the spatial grid, uniformly spaced
xmin = -1.0
xmax = 1.0
nx = 79

# Define the problem>
c = 1.0  # isentropic sound speed
c2 = c**2  # for simplicity below
# timescheme = Euler(delta_t=0.2)
timescheme = RungeKutta3()
cnum = 0.20  # Choose if you fix diffusion number or time step


###########################################################
def reference(x):
    # rho = np.cos(0.5 * np.pi * x) ** 2.0
    rho = np.exp(-(x**2) / 0.01)
    u = np.zeros_like(x)

    return rho, u


def preprocessing():
    # Construct grid (periodic)
    x = np.linspace(xmin, xmax, nx + 1)[:-1]

    # Construct initial state vector (initial condition)
    rho, u = reference(x)

    return x, np.array([rho, u])


def simulation(x, state0):
    # Create a list of checkpointing objects (save a set of times), one per variable
    data = []
    for i in range(np.shape(state0)[0]):
        data.append(checkPointing(delta_t=tcheck))

    # Calculate time step, if needed
    if timescheme.dt == None:
        dt_rho = timeStep(x, velocity=1.0, cnum=cnum)
        dt_u = timeStep(x, velocity=c2, cnum=cnum)
        timescheme.dt = min(dt_rho, dt_u)

    # Integration loop
    t = tmin
    state = np.copy(state0)
    it = 0  # To keep track of how many iterations we need
    for i, item in enumerate(data):
        item.add(it, t, state[i])  # Checkpoint initial condition

    while t < tmax:
        state, t = timescheme.advanceStep(rhs, state, t)  # Advance one time step
        it = it + 1

        if t >= data[0].time2check:  # Checkpoint data
            for i, item in enumerate(data):
                item.add(it, t, state[i])  # Checkpoint initial condition
                item.time2check = min(item.time2check, tmax)  # To always checkpoint last time

    return data


def postprocessing(x, data):
    rho = data[0]  # create views (pointers) for readability
    u = data[1]

    # fig, axs = PlotCurves(x, rho)
    fig, axs, ani = AnimCurves(x, rho)

    cnum = advectionNumber(x, c, timescheme.dt)  # Calculate values with actual delta_t
    axs.set_title(
        r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(
            cnum, np.size(x), rho.ichecked[-1]
        )
    )

    # overwrite default to handle periodic conditions
    axs.set_ylabel("$\\rho$")
    axs.set_xlim(xmin, xmax)
    axs.set_xticks(np.linspace(xmin, xmax, num=5))

    plt.tight_layout(pad=0.1)
    plt.savefig("advection.pdf", bbox_inches="tight")
    plt.show()


def save(x, data):
    rho = data[0]  # define pointers for readability
    u = data[1]

    save_netcdf(
        rho.tchecked,
        [x],
        ["x"],
        [rho.uchecked, u.uchecked],
        ["rho", "u"],
        "acoustics",
    )


###########################################################
h = (xmax - xmin) / nx  # define global variable used below in rhs


def source(x, t):  # periodic source
    s = np.empty_like(x)
    s[:] = 0.0  # no source
    return s


def rhs(state, t):  # Right-hand side of evolution equation (tendency)
    from pynums.fdms.fdm1 import fdm1_e2p

    rho = state[0]  # create views (pointers) for readability
    u = state[1]

    # Using second-order approximations to spatial derivatives
    # Maximum diffusion number is realMax/4
    rho_t = -fdm1_e2p(u) / h
    u_t = -c2 * fdm1_e2p(rho) / h + source(x, t)

    return np.array([rho_t, u_t])


###########################################################
if __name__ == "__main__":
    x, state = preprocessing()
    data = simulation(x, state)
    save(x, data)
    postprocessing(x, data)
