"""Methods and classes for calculations using map coordinates.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path

class GridObject(object):
    """Store and manipulate objects linked to map coordinates.

    Parameters
    ----------
    basemap (Basemap, optional): a basemap object defining the map
    mapparam (dict, optional): the dict defining parameters for a basemap
    shape (tuple, optional): the shape of the grid desired, default (50, 50)

    Note
    ----
    Either basemap or mapparam are needed to instantiate. If both are given,
        basemap is used over mapparam.

    Attributes
    ----------
    x     : 
    y     : 
    shape : 
    """
    def __init__(self, basemap=None, mapparam=None, shape=None):
        if basemap is not None: 
            self.basemap = basemap
        elif mapparam is not None:
            basemap = Basemap(**mapparam)
            self.basemap = basemap
        else:
            raise ValueError('GridObject needs either Basemap object or\
                                paramaters.')

        self.shape = shape or (50, 50)
        self.x = np.linspace(basemap.xmin, basemap.xmax, self.shape[1])
        self.y = np.linspace(basemap.ymin, basemap.ymax, self.shape[0])
        self.Lon, self.Lat = basemap(*np.meshgrid(self.x, self.y), inverse=True)

    def update_shape(self, shape):
        self.shape = shape

        basemap = self.basemap

        self.x = np.linspace(basemap.xmin, basemap.xmax, self.shape[1])
        self.y = np.linspace(basemap.ymin, basemap.ymax, self.shape[0])
        self.Lon, self.Lat = basemap(*np.meshgrid(self.x, self.y), inverse=True)

    def create_interper(self, array):
        """Return a 2D interpolation object on the map, in map coordinates.

        Parameters
        ----------
        array : array to interpolate 

        Returns
        -------
        RectBivariateSpline object

        Notes
        -----
        The interpolation object is defined using map coordinates (because
        these are on a regular grid by necessity (see Basemap docs). To use the
        object, you must convert to map coords with grid.basemap(lons, lats).

        Examples
        --------
        >>> tiltInterp = grid.create_interper(tilt)
        >>> tiltAtPoint = tiltInterp.ev(xp, yp)
        """
        if self.shape != array.T.shape:
            if self.shape == array.shape:
                array = array.T
            else:
                raise ValueError('shapes non compatible with {0}\
                and {1}'.format(self.shape, array.shape))

        return RectBivariateSpline(self.x, self.y, array)

    def interp(self, array, xs, ys, latlon=False):
        """Convenience function for interpolation on the map.

        Parameters
        ----------
        array : array to interp (must be of shape grid.shape)
        xs, ys : the x, y points at which to interpolate the array
        latlon : boolean
            indicates whether xs and ys are given in lat/lon or map coordinates
        
        Examples
        --------
        >>> tiltAtPoints = grid.interp(tilt, lons, lats, latlon=True)
        """
        if latlon: xs, ys = self.basemap(xs, ys)
        interper = self.create_interper(array)
        return interper.ev(xs, ys)

    def selectArea(self, ptlist, latlon=False):
        """Select an area of the grid"""
        ptlist = np.asarray(ptlist)
        if latlon: 
            ptlist[:,0], ptlist[:,1] = self.basemap(ptlist[:,0], ptlist[:,1])
        # create the polygon
        path = Path(ptlist)

        X, Y = np.meshgrid(self.x, self.y)
        # return array indices
        areaind = path.contains_points(zip(X.flatten(), Y.flatten())).reshape(self.shape)
        return areaind



def haversine(lat1, lat2, lon1, lon2, r=6371, radians=False):
    """Calculate the distance bewteen two sets of lat/lon pairs.

    Parameters
    ----------
    lat1, lat2, lon1, lon2 : lat/lon points
    r : radius of sphere
    radians : boolean
        indicates whether points are given in radians (as opposed to degrees)
    """
    if not radians:
        lat1, lat2, lon1, lon2 = np.radians([lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    ds = r*2*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + 
                        np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2))

    return ds
