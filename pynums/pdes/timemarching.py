import numpy as np


# Define time-marching schemes to advance one single step
class Euler:
    realMax = 2.0
    imagMax = 0.0

    def __init__(self, delta_t=None):
        self.dt = delta_t

    def advanceStep(self, rhs, u, t):
        # rhs: function that calculates the rhs of the evolution equation
        # u: vector with state variables
        # t: time
        f = u + self.dt * rhs(u, t)
        return f, t + self.dt


class RungeKutta3:
    realMax = 2.57
    imagMax = 1.73

    def __init__(self, delta_t=None):
        self.dt = delta_t

    def advanceStep(self, rhs, u, t):
        # rhs: function that calculates the rhs of the evolution equation
        # u: vector with state variables
        # t: time
        k1 = rhs(u, t)
        k2 = rhs(u + self.dt * 0.5 * k1, t + 0.5 * self.dt)
        k3 = rhs(u + self.dt * (2.0 * k2 - k1), t + self.dt)
        f = u + self.dt * (k1 + 4.0 * k2 + k3) / 6.0
        return f, t + self.dt


###########################################################
# for advection diffusion problems
def diffusionNumber(x, diffusivity, dt):
    # Calculate minimum grid step
    h = np.min(np.diff(x))

    if diffusivity != None:
        return dt * diffusivity / h**2.0
    else:
        return None


def advectionNumber(x, velocity, dt):
    # Calculate minimum grid step
    h = np.min(np.diff(x))

    if velocity != None:
        return dt * velocity / h
    else:
        return None


def timeStep(x, delta_t=None, diffusivity=None, dnum=None, velocity=None, cnum=None):
    # Variable time step from diffusion and advection stability constraint

    # Calculate minimum grid step
    h = np.min(np.diff(x))

    # We use one over delta_t to deal with cases of diffusivity=0, velocity=0
    if dnum != None and diffusivity != None:
        one_ov_dtd = diffusivity / h**2.0 / dnum
    else:
        one_ov_dtd = 0.0

    if cnum != None and velocity != None:  # Variable time step from advection stability constraint
        one_ov_dtc = velocity / h / cnum
    else:
        one_ov_dtc = 0.0

    one_ov_dt = max(one_ov_dtc, one_ov_dtd)
    if one_ov_dt > 0.0:
        dt = 1.0 / one_ov_dt
    else:
        dt = 1.0 / delta_t

    return dt
