import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
import matplotlib.patches as patches

def iceAreaPlot(ax, areaVerts, basemap, color='k', label=True):
    """Plot ice model area boundaries and labels on a map.

    Labels are placed in the center of the polygonally defined areas.

    Parameters
    ----------
    ax :  <matplotlib.axes>
        The axis to draw on.
    areaVerts : dict
        Dictionary of area definition. Keys are area names, values are Nx2 list
        (or array-like) of lon/lat pairs for area vertices.
    basemap : <mpl_toolkits.basemap.Basemap>
        The basemap object on which to plot.
    color : str
        Color of boundaries and text (Defaults to black).
    label : bool
        Whether to include labels (keys of areaVerts) in centers of areas.
    """    
    # Iterate over the areas
    for key, verts in areaVerts:
        verts = np.asarray(verts)
        # Convert Lon/Lat to basemap coordinates.
        verts = np.array(basemap(verts[:,0], verts[:,1])).T
        # The polygon needs to close, so append first position to end of list.
        verts = np.vstack([verts, verts[0]])
        # Generate an appropriately long list of path actions.
        codes = [Path.MOVETO] + [Path.LINETO for i in range(len(verts)-2)] +\
                [Path.CLOSEPOLY]
        # Create polygon and add it to the axis.
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='None',lw=2, ec='r')
        ax.add_patch(patch)

        if label:
            # First find the center of mass of the polygon (Centroid).
            As = verts[:-1,0]*verts[1:,1] - verts[1:,0]*verts[:-1,1]
            Ai6 = 1./(3*np.sum(As))
            xmid = Ai6*np.sum((verts[:-1,0] + verts[1:,0])*As)
            ymid = Ai6*np.sum((verts[:-1,1] + verts[1:,1])*As)

            # Add the label to center.
            #TODO Check for area text overfill?
            ax.text(xmid, ymid, key, ha='center', va='center',
                fontsize=20, color=color, backgroundcolor=(1,1,1,.9))

    return plt.gca()
