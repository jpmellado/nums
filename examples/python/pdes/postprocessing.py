import numpy as np
import matplotlib.pyplot as plt
from template import *

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

if __name__ != "__main__":
    from __main__ import it
    from __main__ import diffusivity, velocity, dnum, cnum
    from __main__ import reference

def PlotCurves(x, data, reference, filename):
    fig, axs = plt.subplots(1, 1, figsize=figsize11)

    cmap = plt.get_cmap('magma_r')
    legends = []

    for item in range(len(data.tchecked)):
        legends.append(r"Time ${:5.2f}$".format(data.tchecked[item]))
        factor = float(item+1)/float(len(data.tchecked)+1)
        plt.plot(x,data.uchecked[item],color='darkred',alpha=factor)
    #    plt.plot(x,check_usaved[item],color=cmap(factor))

    legends.append(r'Reference')
    axs.plot(x, reference, color='black')

    if diffusivity != None:
        plt.title(r"Diffusion \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(dnum,np.size(x),it))
    if velocity != None:
        plt.title(r"CFL \# ${:3.2f}$. ${:d}$ grid points. ${:d}$ iterations.".format(cnum,np.size(x),it))
        
    plt.xlabel(r"position $x$")
    plt.ylabel(r"function $u$")
    #plt.legend(legends,bbox_to_anchor=(1.02,0.5), loc="center left")
    plt.legend(legends,loc='best')
    plt.grid()

    plt.tight_layout(pad=0.1)
    plt.savefig(filename,bbox_inches='tight')

    plt.show()

    return fig, axs