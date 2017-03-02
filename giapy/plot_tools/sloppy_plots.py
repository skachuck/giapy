import numpy as np
import matplotlib.pyplot as plt

import brewer2mpl

bmap = brewer2mpl.get_map('YlOrRd', 'sequential', 9)
cmap = bmap.get_mpl_colormap(N=1000)

def plotSloppyVisc(s, V, sim=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    if sim is not None:
        labellt = ['%.2f' % x for x in np.log10(sim.earth.u)+21]
        labelrt = ['%.0f' % x for x in sim.earth.d[::-1].cumsum()[::-1]]
    else:
        labellt = labelrt = ['' for x in range(len(s))]

    # Plot the eigenfunctions, add the rule lines
    p = ax.pcolormesh(np.abs(V), cmap=cmap, vmin=0, vmax=1.0)
    ax.hlines(np.arange(n), 0, n, colors='k', lw=0.5, alpha=0.5)
    ax.vlines(np.arange(n), 0, n, colors='w', lw=3)
    
    # left ticks
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(n)+0.5)
    ax.set_xticklabels(['%.2E' % x for x in s], rotation=45, ha='right')
    ax.set_yticks(np.arange(n)+0.5)
    ax.set_yticklabels(labellt)
    ax.set_xlim([0,n])
    ax.set_ylim([0,n])
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Log Viscosity of Section (Pa s)')

    # right ticks
    ax2 = ax.twinx()
    ax2.set_ylim([0,n])
    ax2.set_yticks(np.arange(n)+0.5)
    ax2.set_yticklabels()
    ax2.tick_params('y', left=False, right=True, length=0)
    ax2.set_ylabel('Depth of Section (km)', rotation=270)
    ax2.yaxis.labelpad = 20
    plt.subplots_adjust(bottom=0.2, right=.85, left=0.15)

    return plt.gca()
