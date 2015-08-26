import numpy as np
from scipy.interpolate import interp1d
from progressbar import ProgressBar, Bar, Percentage
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
    
    def getResp(self, t_dur):
        """Return an NDarray (nmax+1, 4) of the responses to a one dyne load applied
        for time t_dur.
        """
        return self.respInterp(t_dur)
    
    def save(self, fname):
        #TODO use dill or klepto packages to serialize interp1d objects
        # interp1d objects can't be pickled, so get rid of them for saving, and
        # reinitialize them afterward.
        self.respInterp = None
        self.params._interpParams = None
        pickle.dump(self, open(fname, 'w'))

        self.params._interpParams = interp1d(self.params.zz,
                                             self.params.paramArray)
        self.respInterp = interp1d(self.times, self.respArray, axis=1)

    def setDesc(self, string):
        self._desc = string 
            
    def calcResponse(self, zarray, nmax=100, nstart=None, prog_track=False):
        """Calculate the response of the Earth to order numbers up to nmax.
        """
        
        if self.nmax is None or self.nmax >= nmax:
            nstart = 1
        else:
            nstart = self.nmax

        respArray = []
        if prog_track:
            pbar = ProgressBar(widgets=['Earth progress: ',  Bar(), Percentage()])
        else:
            pbar = lambda x: x
        for n in pbar(range(nstart, nmax+1)):
            out = self.timeEvolve(n, zarray, nstart)
            respArray.append(out.outArray)

        respArray = np.array(respArray)

        if nstart == 1:
            # Append n=0 zero response
            self.respArray = np.r_[np.zeros((1,out.outArray.shape[0],
                                    out.outArray.shape[1])), respArray]


        self.times = out.times / 3.1536e10  # convert to thousand years
        self.nmax = nmax
        self.respInterp = interp1d(self.times, self.respArray, axis=1)

    def timeEvolve(self, n, zarray, nstart=None):
        out = SphericalEarthOutput()
        
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


def loadEarth(fname):
    earth = pickle.load(open(fname, 'r'))
    earth.params._interpParams = interp1d(earth.params.zz, 
                                          earth.params.paramArray)
    earth.respInterp = interp1d(earth.times, earth.respArray, axis=1)


