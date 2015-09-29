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
        self.x = np.linspace(basemap.xmin, basemap.xmax, self.shape[1],
                                endpoint=False)
        self.y = np.linspace(basemap.ymin, basemap.ymax, self.shape[0],
                                endpoint=True)
        self.Lon, self.Lat = basemap(*np.meshgrid(self.x, self.y), inverse=True)

    def update_shape(self, shape):
        self.shape = shape

        basemap = self.basemap

        self.x = np.linspace(basemap.xmin, basemap.xmax, self.shape[1],
                                endpoint=False)
        self.y = np.linspace(basemap.ymin, basemap.ymax, self.shape[0],
                                endpoint=True)
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
        inds = self.selectArea(area, latlon=latlon, reduced=1)
        dV = self.volume(array)
        return dV[inds].sum()

    def integrateAreas(self, array, areaList):
        """Integrate an array over areas stored in an AreaDict."""
        dV = self.volume(array)
        volList = []
        wholeVol = 0
        for area in areaList:
            inds = self.selectArea(area['vert'], reduced=1)
            volList.append({'name' : area['name'],
                            'vol'  : dV[inds].sum()})
            wholeVol += dV[inds].sum()
        volList.append({'name' : 'whole',
                        'vol'  : wholeVol})
            
        return volList

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

    def selectArea(self, ptlist, latlon=False, reduced=None):
        """Select an area of the grid"""
        ptlist = np.asarray(ptlist)
        if latlon: 
            ptlist[:,0], ptlist[:,1] = self.basemap(ptlist[:,0], ptlist[:,1])
        # create the polygon
        path = Path(ptlist)

        if reduced is not None:
            X, Y = np.meshgrid(self.x[:-reduced], self.y[:-reduced])
            areaind = path.contains_points(zip(X.flatten(), Y.flatten()))
            areaind = areaind.reshape((self.shape[0]-reduced,
                                       self.shape[1]-reduced))
        else:
            X, Y = np.meshgrid(self.x, self.y)
            areaind = path.contains_points(zip(X.flatten(), Y.flatten()))
            areaind = areaind.reshape(self.shape)
        # return array indices
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




def volumeChangeLoad(h, topo):
    """Compute ocean depth changes for a topographic shift h, consistent with
    sloping topographies.
    
    Parameters
    ----------
    h : float
        The topographic shift.
    topo : np.ndarray
        The topography to shift. (Altered topography is T - h)

    Returns
    -------
    hw : np.ndarray
        An array with shape topo.shape whose maximum magnitude is h, with
        decreasing magnitudes along slopes newly submerged or uncovered.
    """

    if h > 0:
        hw = (h - np.maximum(0, topo)) * (h > topo)
    elif h < 0:
        hw = (np.maximum(topo, h)) * (topo < 0)
    else:
        hw = 0*topo

    return hw

def sealevelChangeByMelt(V, topo, grid):
    """Find the topographic lowering that alters the ocean's volume by V.

    Because of changing coastlines, a eustatic increase (decrease) of h will
    generally change the volume of the ocean by more (less) than with a
    'bathtub' ocean model. This function uses a Newton method with an initial
    guess based on the 'bathtub' model.

    Parameters
    ----------
    V : float
        The volume by which to alter the ocean.
    topo : np.ndarray
        The topography to alter.
    grid : <GridObject>
        The grid object assists with integration.
    
    Returns
    -------
    h : float
        The topographic shift consistent with changing / sloping coastlines.
        Note that the new topography after this shift is T - h.
    """
    if V == 0:
        return 0
    # Get first guess of eustatic h.
    h0 = V / grid.integrate(topo < 0, km=False)

    # Use scipy.optimize.root to minimize volume difference.
    Vexcess = lambda h: V - grid.integrate(volumeChangeLoad(h, topo), km=False)
    h = root(Vexcess, h0)

    return h['x'][0]

def oceanUpliftLoad(h, Ta, upl):
    """Compute ocean depth changes for a topographic shift h, consistent with
    sloping topographies.

    Note that the resultant topography is Tb = Ta + upl - h.

    
    Parameters
    ----------
    h : float
        The topographic shift.
    Ta : np.ndarray
        The topography to shift.     
    upl : np.ndarray
        The uplift additionally shifting the topography. Note that uplift and
        geoid affect sea level oppositely (opposite sign).

    Returns
    -------
    hw : np.ndarray
        An array with shape topo.shape whose maximum magnitude is h, with
        decreasing magnitudes along slopes newly submerged or uncovered.

    """
    # The new topography
    Tb = Ta + upl - h
    #               Newly submerged.            Newly emerged.
    hw = (h - upl - np.maximum(Ta, 0))*(Tb<0) + Ta*(Tb>0)*(Ta<0)
    return hw

def sealevelChangeByUplift(upl, topo, grid):
    """Find the topographic lowering that alters the ocean's volume by V.

    Because of changing coastlines, a eustatic increase (decrease) of h will
    generally change the volume of the ocean by more (less) than with a
    'bathtub' ocean model. This function uses a Newton method with an initial
    guess based on the 'bathtub' model.

    Parameters
    ----------
    upl : np.ndarray
        The uplift additionally shifting the topography.Note that uplift and
        geoid affect sea level oppositely (opposite sign).
    topo : np.ndarray
        The topography to alter.
    grid : <GridObject>
        The grid object assists with integration.
    
    Returns
    -------
    h : float
        The topographic shift consistent with changing / sloping coastlines.
        Note that the new topography after this shift is T + upl - h.
        Techincally, upl = uplift - geoid.
    """
    if np.all(upl==0):
        return 0

    # Average ocean floor uplift, for initial guess.
    h0 = grid.integrate(upl*(topo<0), km=False)/grid.integrate(topo<0, km=False)

    # Use scipy.optimize.root to minimize volume difference..
    Vexcess = lambda h: grid.integrate(oceanUpliftLoad(h, topo, upl), km=False)
    h = root(Vexcess, h0)

    return h['x'][0]


def floatingIceRedistribute(I0, I1, S0, grid, denp=0.9077):
    """Calculate load and topographic shift due to ice height changes.

    Calculate the water-equivalent load changes due to changing from ice
    heights I0 to I1 on a solid-surface topography (height of solid earth, NOT
    ice, relative to sea level at t0) S0. The load accounts for the fact that
    when ice is not grounded, it represents a neutral water load, and 
    appropriately updates the 'groundedness' where necessary. 

    The updated solid surface,            S1 = S0 - dhwBar.
    Floating ice can be identified where  (S1 + denp*I1) < 0.
    Topography (to top of ice) is         T1 = S1 + I1*(S1 + denp*I1 >= 0).

    Parameters
    ----------
    I0, I1 : np.ndarrays
        Ice heights (from solid surface to top of ice) at times t0 and t1
    S0 : np.ndarray
        Solid surface topography (height of solid earth relative to sea level)
        at t0.
    grid : <GridObject>
        The grid object assists with integration.
    denp : float
        The ratio of densities of ice and water (default = 0.9077). Used it
        transforming ice heights to equivalent water heights.

    Returns
    -------
    dLoad : np.ndarray
        The total water load induced by ice height chagnes from I0 to I1,
        taking into account floating ice and mass redistribution.
    dhwBar : float
        The topographic shift associated with the mass transfer.
    """
    
    # Find the water-equivalent load change relative to sea level at t0.
    dIwh = (np.maximum(0, S0+denp*I1) - np.maximum(0, S0+denp*I0))*\
        ((S0+denp*I0>=0) + (S0+denp*I1>=0)) # Only where ice for either
                                          # stage are not floating at t0.
                    
    # The change in water volume of the ocean.
    dVo = -grid.integrate(dIwh, km=False)
                                                   
    dhwBar = sealevelChangeByMelt(dVo, S0+denp*I1, grid)
    dLoad = dIwh + volumeChangeLoad(dhwBar, S0+denp*I1)
                                                                                  
    return dLoad, dhwBar
