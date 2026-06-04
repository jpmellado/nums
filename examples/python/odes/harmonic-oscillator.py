import numpy as np
import matplotlib.pyplot as plt
import pynums.ode as odes
from pynums.template import *

# Define the temporal grid
tmin = 0.0  # Initial time
tmax = 2.0  # Final time
num_iterations = 100  # Number of iterations
# tcheck = 0.05  # Time interval to checkpoint data

# Define the problem
period = 1.0  # For clarity
omega = 2.0 * np.pi / period


###########################################################
def reference(t):
    return np.sin(omega * t), omega * np.cos(omega * t)


def preprocessing():
    # Construct initial condition
    X, Y = reference(tmin)

    return [X, Y]


def simulation(state0):
    dt = (tmax - tmin) / num_iterations  # Step size
    t, state = odes.RungeKutta3(rhs, tmin, state0, dt, num_iterations)

    return t, state


def postprocessing(t, u):
    plt.rcParams["axes.spines.top"] = (
        False  # Additional plotting properties for this program
    )
    plt.rcParams["axes.spines.right"] = False

    fig_id = 0
    tag = "harmonic-oscillator"

    fig_id = fig_id + 1
    fig, axs = plt.subplots(2, 1, figsize=figsize11)
    id = 0
    axs[id].plot(t[:], u[:, id])
    axs[id].set_ylabel(r"$X$")
    id = 1
    axs[id].plot(t[:], u[:, id])
    axs[id].set_ylabel(r"$Y$")

    for ax in axs:  # common properties to all axes
        ax.set_xlabel(r"time $t$")
        ax.set_xlim([0, None])
        ax.spines["left"].set_position(("axes", -0.01))
        ax.spines["bottom"].set_position(("axes", -0.01))

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format(tag + "-" + str(fig_id)), bbox_inches="tight")

    #################################################
    fig_id = fig_id + 1
    fig, axs = plt.subplots(1, 1, figsize=figsize11)

    axs.plot(u[:, 0], u[:, 1])
    axs.set_xlabel(r"$X$")
    axs.set_ylabel(r"$Y$")

    plt.tight_layout(pad=0.1)
    plt.savefig("{}.pdf".format(tag + "-" + str(fig_id)), bbox_inches="tight")

    plt.show()


###########################################################
def rhs(state, t):  # Right-hand side of evolution equation (tendency)
    X = state[0]  # create views (pointers) for readability
    Y = state[1]

    dXdt = Y
    dYdt = -(omega**2) * X

    return np.array([dXdt, dYdt])


###########################################################
if __name__ == "__main__":
    state = preprocessing()
    t, state = simulation(state)
    postprocessing(t, state)
