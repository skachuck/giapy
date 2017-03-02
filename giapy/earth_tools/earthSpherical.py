"""
earthSphecidal.y

Author: Samuel B. Kachuck
Date: 

Provides SphericalEarth, a container object for computing, storing, and
retrieving loading response curves on a spherically symmetric earth.
"""
import numpy as np
from scipy.interpolate import interp1d
from giapy import pickle

from giapy.earth_tools.earthIntegrator import SphericalEarthOutput,\
        SphericalEarthShooter,\
        SphericalEarthRelaxer, get_t0_guess, integrateRelaxationDirect,\
        integrateRelaxationScipy

def depthArray(self, npts=30, trunc=True, frac=2/3, n=None, safe=0.9):
    if trunc:
        if n is None: raise ValueError('if truncating, must specify n')
        depth = safe*(1-2*np.pi/(n+0.5))
        nabove = int(frac*npts)
        nbelow = npts-nabove
        zabove = np.linspace(depth, 1, nabove)
        zbelow = np.linspace(self.earth.rCore, depth, nbelow,
                                endpoint=False)
        zarray = np.r_[zbelow, zabove]
    else:
        zarray = np.linspace(self.earth.rCore, 1., npts)
    return zarray

class SphericalEarth(object):
    """A class for calculating, storing, and recalling 

    Stores decay profiles for each spherical order number up to nmax for
        1) ue  : elastic uplift
        2) uv  : viscous uplift
        3) ve  : elastic horizontal
        4) vv  : viscous horizontal
        5) phi1: gravitational potential
        6) g1  : gravtiational acceleration
        7) vel : velocity of uplift

    Note: this object is pickleable - the interp1d object is recreated on load.

    Parameters
    ----------
    params : <giapy.earth_tools.earthParams.EarthParams>
        The Earth parameters to use in this earth model, stored in the giapy
        EarthParams object. See documantation there.

    Methods
    -------
    getResp
    calcResponse
    calcElResponse
    timeEvolve

    Data
    ____
    nmax : int, maximum order number calculated in response curves
    times : ndarray
        The times at which the responses are computed, stored by calcResponse.
    respArray : ndarray, size (nmax+1, len(times), 7), with columns numbered above.
    respInterp : <scipy.interpoate.interp1d>
        An inteprolation objectect computed in SphericalEarth.calcResponse and
        stored for fast retrieval.
    """

    def __init__(self, params):
        self.params = params
        self.nmax = None
        self._desc = ''

    def __repr__(self):
        return self._desc

    def __getstate__(self):
        odict = self.__dict__.copy()
        if getattr(self, 'relaxer', None):
            del odict['relaxer']
        del odict['respInterp']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.respInterp = interp1d(self.times, self.respArray, axis=1)

    
    def getResp(self, t_dur):
        """Interpolate response curves for response to unit load duration t_dur.

        The response curves (stored SphericalEarth.respArray) are interpolated
        lineary for each order number, 0 to SperhicalEarth.nmax) and returned.
        """
        return self.respInterp(t_dur)

    def setDesc(self, string):
        self._desc = string 
            
    def calcResponse(self, zarray, nmax=100, nstart=None, times=None):
        """Calculate the response of the Earth to order numbers up to nmax.
        """
        
        if self.nmax is None or self.nmax >= nmax:
            nstart = 1
        else:
            nstart = self.nmax

        respArray = []
        for n in range(nstart, nmax+1):
            out = self.timeEvolve(n, zarray, nstart, times)
            respArray.append(out.outArray)

        respArray = np.array(respArray)

        if nstart == 1:
            # Append n=0 zero response
            self.respArray = np.r_[np.zeros((1,out.outArray.shape[0],
                                    out.outArray.shape[1])), respArray]
        else:
            self.respArray = np.r_[self.respArray, respArray]


        self.times = out.times / 3.1536e10  # convert back to thousand years
        self.nmax = nmax
        self.respInterp = interp1d(self.times, self.respArray, axis=1)

    def calcElResponse(self, zarray, nmax):
        """Compute the elastic loading response of the Earth for order numbers
        between nstart and nmax.

        Parameters
        ----------
        zarray : ndarray. The array of depths to use in relaxation method.
        nmax   : int. The highest order number to compute.

        Returns
        -------
        respArray : ndarray, shape (nmax, 5)
            An array of instananeous elastic responses to order numbers 1 to
            nmax. The columns are ue, ve, phi1, g1, vel (see class
            documentation for description).
        """
        
        respArray = []
        yE0, yV0 = get_t0_guess(self.params, zarray, n=1)
        # Use the relaxer to get subsequent order numbers
        self.relaxer = SphericalEarthRelaxer(self.params,
                            zarray, yE0, yV0, 1)

        disps = np.zeros_like(zarray)
        for n in range(1, nmax+1):
            self.relaxer.changeOrder(n)
            self.relaxer(0, disps)
            respArray.append(self.relaxer.solout())
        return np.array(respArray)

    def timeEvolve(self, n, zarray, nstart=None, times=None):
        out = SphericalEarthOutput(times)
        
        if n == nstart or nstart is None:
            # For the first one, set up the relaxer and initial guess.
            yE0, yV0 = get_t0_guess(self.params, zarray, n=n)
            self.relaxer = SphericalEarthRelaxer(self.params, 
                                zarray, yE0, yV0, n)
            integrateRelaxationScipy(self.relaxer, out) 
        else:
            # Subsequent calculations use n-1, t~0 as initial guess.
            self.relaxer.changeOrder(n)
            try:
                integrateRelaxationScipy(self.relaxer, out)
            except:
                # If Solvde takes too many steps, use shooting method to
                # generate new guess and continue.
                print 'Reguessing: n={0}'.format(n)
                yE0, yV0 = get_t0_guess(self.params, zarray, n=n)
                self.relaxer = SphericalEarthRelaxer(self.params, 
                                    zarray, yE0, yV0, n)
                integrateRelaxationScipy(self.relaxer, out)
        return out
