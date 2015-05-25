"""
map_tools.py

    Methods and classes for calculations using map coordinates.

    Author: Samuel B. Kachuck
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
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

    def volume(self, array, km=True):
        """Weight an area defined over the map by the area of the cells
        """
        try:
            x = array.shape
        except:
            raise ValueError('provided array has no "shape" property')
        if self.shape != array.shape:
            raise ValueError('GridObject and array must have same shape')

        dLon = np.abs(self.Lon[:-1,1:]-self.Lon[:-1,:-1])*np.pi/180
        dLat = np.abs(self.Lat[1:,:-1]-self.Lat[:-1,:-1])*np.pi/180
        
        r = 6371 if km else 6371000

        # formula needs colatitude
        CoLat = self.Lat+90
        dA = (r**2)*np.sin(CoLat[:-1, :-1]*np.pi/180)*dLat*dLon
        dV = array[:-1,:-1]*dA

        return dV

    def integrate(self, array, km=True):
        """Perform area integration of an array over the map area.
        """
        dV = self.volume(array, km)
        return dV.sum()

    def integrateArea(self, array, area, latlon=False):
        """Integrate an array over a specific area."""
        inds = self.selectArea(area, latlon=latlon)
        dV = self.volume(array)
        return dV[inds].sum()

    def integrateAreas(self, array, areaDict):
        """Integrate an array over areas stored in an AreaDict."""
        dV = self.volume(array)
        volDict = {'whole': dV.sum()}
        for name, area in areaDict.iteritems():
            inds = self.selectArea(area['verts'], latlon=area['latlon'])
            volDict[name] = dV[inds].sum()
            
        return volDict

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

    def pcolormesh(self, Z, **kwargs):
        p = self.basemap.pcolormesh(self.Lon, self.Lat, Z, **kwargs)
        return p 



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

def loadXYZGridData(fname, shape=None, lonlat=False, **kwargs):
    """Load data on an evenly spaced grid from an XYZ format.

    Parameters
    ----------
    fname : file or str
        The path to the file to be loaded
    shape : tuple
        The shape of the grid. Default is square grid.
    lonlat : boolean
        Return lon, lat, data if True (default False).
    **kwargs : see np.loadtxt documentation.
    """
    rawData = np.loadtxt(fname, **kwargs)

    if len(rawData.shape) == 1:
        XY = False
    else:
        XY = True
    
    if shape is not None:
        nx, ny = shape[0], shape[1]
        shape = (3, nx, ny) if XY else (nx, ny)
    else:
        n = np.sqrt(rawData.shape[0])
        shape = (3, n, n) if XY else (n, n)

    if not XY:
        return rawData.reshape(shape)
    elif lonlat:
        return rawData.T.reshape(shape)
    else:
        return rawData.T.reshape(shape)[2]




def eustaticChange(h, topo, upl=0):
    """Computes ocean depth changes for eustatic change h, consistent with
    sloping topographies."""

    if h > 0:
        hw = (h - np.maximum(0, topo) - upl) * (h > topo + upl)
    elif h < 0:
        hw = (np.maximum(topo, h) - upl) * (topo + upl < 0)
    else:
        raise ValueError('h cannot be equal to 0')

    return hw

def eustaticChangeByVolume(V, topo, grid, upl=0):
    """Find the eustatic change that alters the ocean's volume by V.

    Because of changing coastlines, a eustatic increase (decrease) of h will
    generally change the volume of the ocean by more (less) than with a
    'bathtub' ocean model. This function uses a Newton method with an initial
    guess based on the 'bathtub' model.
    """
    if V == 0:
        return 0
    # Get first guess of eustatic h.
    h0 = V / grid.integrate(topo < 0, km=False)

    diff = lambda h: V - grid.integrate(eustaticChange(h, topo, upl), km=False)
    h = root(diff, h0)

    return h['x'][0]

#def rectifyMassBalance(wLoad0, wLoad1, topo, grid):
#    """Calculate the ocean load change associated with a continental ice
#    change.
#    """
#    # Calculate the current eustatic level
#    Vequiv0 = grid.integrate(wLoad0, km=False)
#    # 74 accounts for ESL equivalent of remaining present day ice.
#    #TODO Generalize this.
#    he = eustaticChangeByVolume(-Vequiv0, topo, grid) + 74
#    if he == 0:
#        hw = 0
#    else:
#        hw = eustaticChange(he, topo)
#
#    dLoad = wLoad1-wLoad0
#    # Calculate the water equivalent volume of the ice change.
#    dVequiv = grid.integrate(dLoad, km=False)
#    # Calculate the eustatic change, consistent with changing shorelines.
#    dhe = eustaticChangeByVolume(-dVequiv, topo-hw, grid)
#    if dhe == 0:
#        return dLoad
#    # Get the ocean load of that eustatic change.
#    dhw = eustaticChange(dhe, topo-hw)
#    # Add it to the load and return.
#    return dLoad + dhw

def rectifyMassBalance(wLoad0, wLoad1, upl0, upl1, topo, grid):
    # Calculate topography and esl at t0.
    topop = topo + upl0 + wLoad0            # Topographic changes.
    Vequiv0 = grid.integrate(wLoad0, km=False)
    he = eustaticChangeByVolume(-Vequiv0, topop, grid) + 74
    if he == 0:
        hw = 0
    else:
        hw = eustaticChange(he, topop)
    topop -= hw                             # Topography refelcts eustasy.

    dLoad = wLoad1 - wLoad0
    dUpl = upl1 - upl0
    dVequiv = grid.integrate(dLoad, km=False)
    dhe = eustaticChangeByVolume(-dVequiv, topop, grid, dUpl)
    if dhe == 0:
        return dLoad
    dhw = eustaticChange(dhe, topop, dUpl)
    return dLoad + dhw

