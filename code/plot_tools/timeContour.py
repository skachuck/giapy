import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import brewer2mpl

bmap = brewer2mpl.get_map('YlGnBu', 'sequential', 9)
cmap = bmap.get_mpl_colormap(N=10)

mappable = ScalarMappable(cmap=cmap)

def timeContour(ax, X, Y, data, times, levels):
    """Plot contours over time.
    """
    try:
        if len(levels)==1 and len(times)>1:
            levels = np.repeat(levels, len(times))
    except:
        levels = np.repeta(levels, len(times))
    colors = cmap((times-times.min())/(times.max()-times.min()))
    for color_t, data_t, level in zip(colors, data, levels):
        ax.contour(X, Y, data_t, levels=[level], colors=[color_t])
    #mappable.set_array(times)
    #plt.colorbar(mappable)
    return plt.gca()
