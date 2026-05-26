import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from template import *

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.fontsize"] = "small"


def PlotCurves(x, data):
    fig, axs = plt.subplots(1, 1, figsize=figsize11)

    for item in range(len(data.tchecked)):
        factor = float(item + 1) / float(len(data.tchecked) + 1)
        plt.plot(
            x,
            data.uchecked[item],
            color="darkred",
            alpha=factor,
            label=r"time $t={:5.2f}$".format(data.tchecked[item]),
            clip_on=False,
        )

    axs.spines["left"].set_position(("axes", -0.03))
    axs.spines["bottom"].set_position(("axes", -0.03))
    # axs.xaxis.set_major_locator(plt.MaxNLocator(5))
    axs.set_xlim([x[0], x[-1]])
    axs.set_xticks(np.linspace(x[0], x[-1], num=5))
    axs.set_xlabel(r"position $x$")
    axs.set_ylabel(r"function $u$")
    plt.legend(loc="best")
    plt.grid()

    return fig, axs


def PlotContours(grid, data, levels):
    xc = grid[0]  # for clarity
    yc = grid[1]

    fig, axs = plt.subplots(1, 1, figsize=figsize11)
    axs.spines["left"].set_position(("axes", -0.03))
    axs.spines["bottom"].set_position(("axes", -0.03))
    axs.set_xlabel(r"position $x$")
    axs.set_ylabel(r"position $y$")

    for item in range(len(data.tchecked)):
        factor = float(item + 1) / float(len(data.tchecked) + 1)
        c = plt.contour(
            xc, yc, data.uchecked[item], levels=levels, colors="darkred", alpha=factor
        )
        # c = plt.pcolormesh(xc, yc, data.uchecked[item])
        # plt.colorbar(c) #, label='field')

    return fig, axs


def AnimContours(grid, data):
    xc = grid[0]  # for clarity
    yc = grid[1]

    fig, axs = plt.subplots(1, 1, figsize=figsize11)
    axs.spines["left"].set_position(("axes", -0.03))
    axs.spines["bottom"].set_position(("axes", -0.03))
    axs.set_xlabel(r"position $x$")
    axs.set_ylabel(r"position $y$")

    cs = []
    for item in range(len(data.tchecked)):
        c = plt.pcolormesh(xc, yc, data.uchecked[item], animated=True)
        if item == 0:
            plt.pcolormesh(xc, yc, data.uchecked[item])
        cs.append([c])

    ani = animation.ArtistAnimation(fig, cs, interval=50, blit=True, repeat_delay=1000)

    return fig, axs, ani
