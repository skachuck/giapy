import numpy as np
import matplotlib.pyplot as plt

def plotViscDecay(earth, ax=None, xlim=[1e-5,4e-4], ylim=[0,12], **kw):
    if ax is None: fig, ax = plt.subplots(1,1)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for n, nDecay in enumerate(earth.respArray[1:], start=1):
        ax.plot(np.abs(nDecay[:,1]), earth.times, **kw)

    return plt.gca()
