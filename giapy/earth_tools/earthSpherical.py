import numpy as np
from scipy.interpolate import interp1d
import cPickle as pickle

from .earthIntegrator import SphericalEarthOutput, SphericalEarthShooter,\
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
        1) Surface Uplift
        2) Elastic Uplift
        3) Viscous Uplift Rate
        4) Gravitational potential
        5) Geoid
    responses to a unit load.

    ntrunc : int
        maximum spherical order number
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
        """Return an NDarray (nmax+1, 4) of the responses to a one dyne load applied
        for time t_dur.
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
