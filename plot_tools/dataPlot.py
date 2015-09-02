import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

def plotStdErrorsOnMap(lons, lats, ses, numPts=None, basemap=None, ax=None):
    basemap = basemap or Basemap()
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(15,10))

    if numPts is None:
        s = 50
    else:
        normPts = numPts/float(max(numPts))
        s = 500*normPts

    datamax = np.max(np.abs(ses))
    datamag = np.floor(np.log10(datamax))
    vmax = (10**datamag)*np.maximum(
            np.floor(datamax/(10**datamag)/2), 1.)
    vmin=-vmax


    basemap.drawcoastlines(color=(1,1,1,1), ax=ax, zorder=0)
    xs, ys = basemap(lons, lats)
    p = basemap.scatter(xs, ys, c=ses, s=s, 
                        vmin=vmin, vmax=vmax, cmap='RdYlBu_r', 
                        ax=ax, edgecolor='None', alpha=0.5)
    ax.set_axis_bgcolor((0,0,0,0.2))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    plt.colorbar(p, cax=cax, label='Standard Error (m)')

    # If scaling point size by number of points, add a label.
    if numPts is not None:
        samplept = int(5*10**(np.floor(np.log10(max(numPts)))-1))
        basemap.scatter([0.05], [0.1], c='k', alpha=0.75, 
                        s=500*samplept/float(max(numPts)),
                        vmin=vmin, vmax=vmax, ax=ax,
                        edgecolor='None', transform=ax.transAxes)
        ax.text(0.065, 0.1, '- {0:d} observations at site'.format(samplept),
            transform=ax.transAxes, va='center', fontsize=12)

    return plt.gca()

def plotLocTimeseries(data, calc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.plot(data.ts, data.ys, marker='+', color='k', alpha=0.75,
            ms=15, ls='None')
    ax.plot(calc.ts, calc.ys)
    ax.set_title(str(data))
    return plt.gca()
