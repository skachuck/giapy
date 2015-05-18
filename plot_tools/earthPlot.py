import numpy as np
import matplotlib.pyplot as plt

def plotViscDecay(earth, ax=None, nmin=1, nmax=-1, nskip=1, 
                    xlim=[1e-5,4e-4], ylim=[0,12], **kw):
    if ax is None: fig, ax = plt.subplots(1,1)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for n, nDecay in enumerate(earth.respArray[nmin:nmax:nskip], start=nmin):
        ax.plot(np.abs(nDecay[:,1]), earth.times, **kw)

    return plt.gca()
