import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from giapy.map_tools import haversine

class TransectPath(object):
    """Simple container for a transects xs, ys, and distances along, ds"""
    def __init__(self, xs, ys, ds):
        if not len(xs)==len(ys)==len(ds):
            raise TypeError('xs, ys, and ds must have same length')
        self.xs = xs
        self.ys = ys
        self.ds = ds
        self.n = len(self.xs)

def linear_path(p0, p1, n=50):
    """Return a list of n coordinate pairs along a linear path from p0 to p1"""
    m = float(p1[1]-p0[1])/float(p1[0]-p0[0])    
    xs = np.linspace(p0[0], p1[0], n)
    ys = p0[1]+m*(xs-p0[0])    
    ds = np.sqrt((xs-p0[0])**2 + (ys-p0[1])**2)
    
    return TransectPath(xs, ys, ds)
    
def sphere_path(p0, p1, n=50, r=6371):
    """Return a list of n equally spaced (lon, lat) pairs along a path from p0 
    to p1, where p0 and p1 are given in (lon, lat)."""
    
    m = float(p1[1]-p0[1])/float(p1[0]-p0[0])    
    lons = np.linspace(p0[0], p1[0], n)
    lats = p0[1]+m*(lons-p0[0])
    
    # Now get the distances along that line
    dds = np.zeros(n)
    dds[1:] = haversine(lats[:-1], lats[1:], lons[:-1], lons[1:])
    
    # Sum up distances along the path
    ds = dds        
    for i, d in enumerate(dds):
        ds[i] = d + ds[max(i-1, 0)]
    
    return TransectPath(lons, lats, ds)
    
def map_path(p0, p1, m, n=50, lonlat=False, r=None):
    # get the lons and lats along a great circle between p0 and p1 that are
    # equally spaced on the projection of m
    lons, lats = m(*m.gcpoints(p0[0], p0[1], p1[0], p1[1], n), inverse=True)
    
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    # Now get the distances along that line
    r = r or m.rmajor
    dds = np.zeros(n)
    dds[1:] = haversine(lats[:-1], lats[1:], lons[:-1], lons[1:], r)
    
    # Sum up distances along the path
    ds = dds 
    for i, d in enumerate(dds):
        ds[i] = d + ds[max(i-1, 0)]
    
    if lonlat:
        return TransectPath(lons, lats, ds)
    else:
        xs, ys = m(lons, lats)
        return TransectPath(xs, ys, ds)

def off_grid_profile(x, y, Z, path, interper=None):
    transector = interper or RectBivariateSpline(x, y, Z.T)
    transect = transector.ev(path.xs, path.ys)
    
    return transect

def transect_with_inset_context(X, Y, Z, **kw):
    if 'p0' and 'p1' and 'n' in kw.keys():
        path = linear_path(kw['p0'], kw['p1'], kw['n'])
    elif 'path' and 'ds' in kw.keys():
        path = kw['path']
    else:
        raise NameError
    
    if np.size(np.shape(X))>1:
        x, y = X[0,:], Y[:,0]
    else:
        x, y = X, Y
        X, Y = np.meshgrid(x, y)    
        
    transect = off_grid_profile(x, y, Z, path)
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(path.ds, transect)
    insetax = fig.add_axes([0.15, 0.60, 0.25, 0.25])
    insetax.contourf(X, Y, Z)
    insetax.plot(path.xs, path.ys)

def point2line(xp, x0, x1):
    """Project points xp onto a line defined by endpoints x0 and x1
    """
    l = x1-x0
    lhat = l/np.sqrt(l.dot(l))
    r = xp-x0
    xc = x0+(r[:, np.newaxis].dot(lhat))*lhat
    return xc
