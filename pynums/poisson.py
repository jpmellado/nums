import numpy as np
from pynums.fdms.bvp2 import *


def poisson_periodic(grid, s, bcs):
    x = grid[0]  # for clarity
    y = grid[1]
    bcs_b = bcs[0]  # function defining bottom boundary conditions
    bcs_t = bcs[1]  # function defining top boundary conditions

    nx = np.size(x[0, :])
    hx = x[0, 1] - x[0, 0]  # assuming uniform grid
    hy = y[1, 0] - y[0, 0]

    source = s(x, y)  # source term
    fs = np.fft.rfft(source)

    u1 = bcs_b(x[0, :])  # boundary condition at the bottom
    fu1 = np.fft.rfft(u1)

    un = bcs_t(x[0, :])  # boundary conditions at the top
    fun = np.fft.rfft(un)

    # solve fft x-transform system for each mode
    fu = np.empty_like(fs)

    w0 = 2.0 * np.pi / nx / hx

    ki = 0
    fu[:, ki] = bvp2_e121_dn(fs[:, ki] * hy**2, 0.0, hy * fun[ki])

    for ki in range(1, np.size(fs, 1)):
        fu[:, ki] = bvp2extended_e121_nn(
            fs[:, ki] * hy**2, (ki * w0) ** 2 * hy**2, hy * fu1[ki], hy * fun[ki]
        )

    u = np.fft.irfft(fu)

    return u
