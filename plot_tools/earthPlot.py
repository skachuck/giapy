import numpy as np
import matplotlib.pyplot as plt

def plotViscDecay(earth, ax=None, nmin=1, nmax=-1, nskip=1, 
                    xlim=[1e-5,5e-4], ylim=[0,12], **kw):
    if ax is None: fig, ax = plt.subplots(1,1)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    paramSurf = earth.params.getParams(1.)
    rhog = paramSurf['den']*paramSurf['grav']
    if nmax == -1: nmax = earth.respArray.shape[0]
    ns = np.mgrid[nmin:nmax:nskip]
    for n, nDecay in zip(ns, earth.respArray[ns]):
        vislim = 1./(rhog * earth.params.getLithFilter(n=n))
        ax.plot(nDecay[:,1]+vislim, earth.times, **kw)

    return plt.gca()
