import numpy as np
import matplotlib.pyplot as plt

def plotViscDecay(earth, ax=None, nmin=1, nmax=-1, nskip=1, 
                    ns=None, xlim=[1e-5,5e-4], ylim=[0,12], **kw):
    if ax is None: fig, ax = plt.subplots(1,1)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    paramSurf = earth.params.getParams(1.)
    rhog = paramSurf['den']*paramSurf['grav']
    if nmax == -1: nmax = earth.respArray.shape[0]
    ns = ns or np.mgrid[nmin:nmax:nskip]
    for n, nDecay in zip(ns, earth.respArray[ns]):
        vislim = 1./(rhog * earth.params.getLithFilter(n=n))
        ax.plot(nDecay[:,1]+vislim, earth.times, **kw)

    return plt.gca()

def plotProfiles(zarray, yEyV, axs=None, **kw):
    if axs is None: fig, axs = plt.subplots(2, 5, figsize=(15,6),
                                            sharex=True)

    titles = ['$U_E$', '$V_E$', '$P_E$', '$Q_E$', '$\phi1$', 
              '$g1$' , '$U_V$', '$V_V$', '$P_V$', '$Q_V$'   ]

    for ax, prof, title in zip(axs.flatten(), yEyV, titles):
        ax.plot(zarray, prof, **kw)
        ax.set_title(title)
    
    return plt.gca()
