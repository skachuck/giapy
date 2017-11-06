"""
earthSphericialLap.py

Author: Samuel B. Kachuck
Date: Oct 9, 2017

Provides SphericalEarthLap, a container object for reading, storing, and
retrieving loading response curves on a spherically symmetric earth in the
Laplace domain.
"""

import numpy as np
from giapy.giasim import AbstractEarthGiaSimObserver

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

    def __init__(self): 
        self.nmax = None
        self._desc = ''

    def __repr__(self):
        return self._desc
    
    def getResp(self, ts):
        """Interpolate response curves for response to unit load duration t_dur.

        The response curves (stored SphericalEarth.respArray) are interpolated
        lineary for each order number, 0 to SperhicalEarth.nmax) and returned.
        """

        ts = np.atleast_1d(ts)
        # 
        
        resp = self.hlke[...,None]+np.einsum('ijk,ijl->ikl', self.hlks[:,:,1:],
                        (1.-np.exp(np.multiply.outer(self.hlks[:,:,0], ts))))

        return np.squeeze(resp)

    def setDesc(self, string):
        self._desc = string 
            
    def loadLoveNumbers(self, fname, drctry='./'):
        """Calculate the response of the Earth to order numbers up to nmax.
        """
        
        text = np.loadtxt(drctry+fname)

        nmin = 1
        nmax = int(text[-1][0])

        # Preallocate memory
        hlke = np.zeros((nmax+1, 3))
        hlkf = np.zeros((nmax+1, 3))
        hlks = np.zeros((nmax+1, 9, 4))
        

        # Find end of earth model
        i = 1
        while text[i][0] != 1:
            i += 1

        i = 5
        j = 1
        while True:
            try: 
                line = text[i]
            except:
                break
            n, nlines, k, h, l = line
            hlke[j] = h, l, k
            i += 1
            for m in range(int(nlines)):
                hlks[j, m, :] = text[i][[1, 3, 4, 2]]
                i += 1
            hlkf[j] = text[i][[3, 4, 2]]
            i+=1
            j += 1


        self.nmax = nmax
        self.hlke, self.hlkf, self.hlks = hlke, hlkf, hlks

    def loadTabooNumbers(self, drctry='./'):
        hef = np.loadtxt(drctry+'h.dat', skiprows=2)
        kef = np.loadtxt(drctry+'k.dat', skiprows=2)
        lef = np.loadtxt(drctry+'l.dat', skiprows=2)
        ss = np.loadtxt(drctry+'spectrum.dat', skiprows=7, comments='>')
        hs = np.loadtxt(drctry+'ih.dat', skiprows=2, comments='>')
        ks = np.loadtxt(drctry+'ik.dat', skiprows=2, comments='>')
        ls = np.loadtxt(drctry+'il.dat', skiprows=2, comments='>')
        
        nmax = hef.shape[0]
        ns = hef.shape[1] - 3

        hlke = np.zeros((nmax+1, 3))
        hlkf = np.zeros((nmax+1, 3))
        hlke[1:] = np.vstack([hef[:,1], lef[:,1], kef[:,1]]).T
        hlkf[1:] = np.vstack([hef[:,2], lef[:,2], kef[:,2]]).T

        hlks = np.zeros((nmax+1, ns, 4))
        hlks[1:,:,0] = ss[:,2].reshape(nmax,ns)
        hlks[1:,:,1] = hs.reshape(nmax,ns,2)[:,:,1]
        hlks[1:,:,2] = ls.reshape(nmax,ns,2)[:,:,1]
        hlks[1:,:,3] = ks.reshape(nmax,ns,2)[:,:,1]

        self.nmax = nmax
        self.hlke, self.hlkf, self.hlks = hlke, hlkf, hlks

    
    class TotalUpliftObserver(AbstractEarthGiaSimObserver):
        def isolateRespArray(self, respArray):
            RE = 6371000.
            ME = 5.972e24
            GSURF = 9.815
            psi_l = 4*np.pi*6.674e-11*RE/(2*self.ns+1.)
            psi_l = 4*np.pi*RE**3/(2*self.ns+1.)/ME
            return respArray[self.ns,0]*psi_l
    
    
    class TotalHorizontalObserver(AbstractEarthGiaSimObserver):
        def isolateRespArray(self, respArray):
            RE = 6371000.
            ME = 5.972e24
            GSURF = 9.815
            psi_l = 4*np.pi*6.674e-11*RE/(2*self.ns+1.)
            psi_l = 4*np.pi*RE**4/(2*self.ns+1.)/ME
            return respArray[self.ns,1]*psi_l
    
        def transform(self, trans):      
            u, v = trans.getgrad(self.array)
            return u, v
    
    class GeoidObserver(AbstractEarthGiaSimObserver):
        def isolateRespArray(self, respArray):
            # Divide the negative potential by PREM surface gravity,
            # 982.22 cm/s^2, to get the geoid shift. (negative because when the
            # potential at the surface decreases, the equipotential surface
            # representing the ocean must have risen.)
            #TODO make this a not hard-coded number (do in earth model?)
            # 1e-2 makes the response in m displacement / dyne ice
            RE = 6371e3
            ME = 5.972e24
            GSURF = 9.815
            psi_l = 4*np.pi*6.674e-11*RE/(2*self.ns+1.)/GSURF
            psi_l = 4*np.pi*RE**3/(2*self.ns+1.)/ME
            return (1+respArray[self.ns,2])*psi_l
    
    class GravObserver(AbstractEarthGiaSimObserver):
        def isolateRespArray(self, respArray):
            return 0
    
    class VelObserver(AbstractEarthGiaSimObserver):
        def isolateRespArray(self, respArray):
            return 0
    
    class MOIObserver(AbstractEarthGiaSimObserver):
        pass
    
    class AngularMomentumObserver(AbstractEarthGiaSimObserver):
        pass
