import numpy as np
import matplotlib.pyplot as plt
from template import *

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.fontsize"] = "small"


def PlotCurves(x, data):
    fig, axs = plt.subplots(1, 1, figsize=figsize11)

    cmap = plt.get_cmap("magma_r")

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
