import numpy as np
import matplotlib.pyplot as plt
from template import *

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


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
        )

    axs.set_xlabel(r"position $x$")
    axs.set_ylabel(r"function $u$")
    plt.legend(loc="best")
    plt.grid()

    return fig, axs
