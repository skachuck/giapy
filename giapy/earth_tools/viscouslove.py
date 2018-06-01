"""
viscouslove.py
Author: Samuel B. Kachuck
Date: July 13, 2017

    Compute decoupled viscous love numbers.

    Note on gravity perturbation. This code supports two definitions of the
    gravity perturbation, using the keyword Q. Q=1 is simply the radial
    derivative of the perturbation of the gravtiational potential. Q=2
    corresponds to the generalized flux, which is defined 
    $$Q_2=4\pi G U_L+\frac{\ell+1}{r}\Psi+\frac{\partial \Psi}{\partial r}.$$
"""

from __future__ import division
import numpy as np
from giapy.numTools.solvdeJit import interior_smatrix_fast
# Check for numba, use if present otherwise, skip.
try:
    from giapy.numTools.solvdeJit import interior_smatrix_fast, solvde
    from numba import jit, void, int64, float64
    numba_load = True
except ImportError:
    from giapy.numTools.solvde import interior_smatrix_fast, solvde
    numba_load = False

def propMatVisc(zarray, n, params, t=1, Q=1, scaled=False, logtime=False):
    """Generate the viscous love-number propagator matrix for zarray. 
    
    Inhomogeneities from elastic deformation and viscous migration of 
    nonadiabatic density gradients are added using gen_viscb.

    Parameters
    ----------
    zarray : numpy.ndarray or float
        The (normalized) radii at which to compute the propagators. May be
        singleton.
    n : int. The spherical harmonic order number.
    params : <giapy.earth_tools.earthParams.EarthParams>
    Q : int. Gravity perturbation definition flag (see note at top of file).

    Returns
    -------
    a : numpy.ndarray of the propagator matrix. Has shape (len(zarray), 6, 6) 
        or (4,4).
    """
    assert params.normmode == 'love', 'Must normalize parameters'
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        singz = True
    
    parvals = params.getParams(zarray)

    eta = parvals['visc']

    z_i = 1./zarray

    a = np.zeros((len(zarray), 4, 4))

    if not scaled:
        if logtime:
            fillfunc = _matFilllog
        else:
            fillfunc = _matFill
    else:
        if logtime:
            fillfunc = _matFilllogscale 
        else:
            fillfunc = _matFillscale

    # Fill the matrix. It's a long for loop that can be accelerated with numba.
    fillfunc(a, n, zarray, eta, z_i, 2*(n+1), 1./(2*n+1),
                        params.getLithFilter(n=n), t)

    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T

def _matFill(a, n, zarray, eta, z_i, l, li, alpha, t): 
    for i in range(len(zarray)):
        
        # r d\dt{h}/dr
        a[i,0,0] = -2
        a[i,0,1] = n+1
        a[i,0,2] = 0#(2*n+1)*zarray[i]/eta[i]
        
        # r d\dt{L}/dr
        a[i,1,0] = -n
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = zarray[i]/eta[i]*l*l/alpha
        
        # r df_L/dr 
        a[i,2,0] = 12*eta[i]*z_i[i]*li*li*alpha
        a[i,2,1] = -6*eta[i]*z_i[i]*(n+1)*li*li*alpha
        a[i,2,2] = 0.
        a[i,2,3] = n+1
        
        # r dF_M/dr
        a[i,3,0] = -6*eta[i]*z_i[i]*n*li*li*alpha
        a[i,3,1] = 2*eta[i]*z_i[i]*(2*n*(n+1)-1)*li*li*alpha
        a[i,3,2] = -n
        a[i,3,3] = -3

def _matFillscale(a, n, zarray, eta, z_i, l, li, alpha, t):
    for i in range(len(zarray)):
        
        # r d\dt{h}/dr
        a[i,0,0] = -4*li
        a[i,0,1] = 2*(n+1)*li 
        
        # r d\dt{L}/dr
        a[i,1,0] = -2*n*li
        a[i,1,1] = 2*li
        a[i,1,2] = 0.
        a[i,1,3] = 2*zarray[i]/eta[i]*l/alpha
        
        # r df_L/dr 
        a[i,2,0] = 24*eta[i]*z_i[i]*li*li*li*alpha
        a[i,2,1] = -12*eta[i]*z_i[i]*(n+1)*li*li*li*alpha
        a[i,2,2] = 0.
        a[i,2,3] = 2*(n+1)*li
        
        # r dF_M/dr
        a[i,3,0] = -12*eta[i]*z_i[i]*n*li*li*li*alpha
        a[i,3,1] = 4*eta[i]*z_i[i]*(2*n*(n+1) - 1)*li*li*li*alpha
        a[i,3,2] = -2*n*li
        a[i,3,3] = -6*li

def _matFilllog(a, n, zarray, eta, z_i, l, li, alpha, t): 
    for i in range(len(zarray)):
        
        # r d\dt{h}/dr
        a[i,0,0] = -2
        a[i,0,1] = n+1
        a[i,0,2] = 0#(2*n+1)*zarray[i]/eta[i]
        
        # r d\dt{L}/dr
        a[i,1,0] = -n
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = -zarray[i]*l/eta[i]/t/alpha
        
        # r df_L/dr 
        a[i,2,0] = -12*eta[i]*z_i[i]*li*t*alpha
        a[i,2,1] = 6*eta[i]*z_i[i]*(n+1)*li*t*alpha
        a[i,2,2] = 0.
        a[i,2,3] = n+1
        
        # r dF_M/dr
        a[i,3,0] = 6*eta[i]*z_i[i]*n*li*t*alpha
        a[i,3,1] = -2*eta[i]*z_i[i]*(2*n*(n+1) - 1)*li*t*alpha
        a[i,3,2] = -n
        a[i,3,3] = -3

def _matFilllogscale(a, n, zarray, eta, z_i, l, li, alpha, t):
    for i in range(len(zarray)):
        
        # r d\dt{h}/dr
        a[i,0,0] = -4*li
        a[i,0,1] = 2*(n+1)*li 
        
        # r d\dt{L}/dr
        a[i,1,0] = -2*n*li
        a[i,1,1] = 2*li
        a[i,1,2] = 0.
        a[i,1,3] = 2*zarray[i]/eta[i]/t/alpha

        # r df_L/dr 
        a[i,2,0] = -24*eta[i]*z_i[i]*li*li*li*t*alpha
        a[i,2,1] = 12*eta[i]*z_i[i]*(n+1)*li*li*li*t*alpha
        a[i,2,2] = 0.
        a[i,2,3] = 2*(n+1)*li
        
        # r dF_M/dr
        a[i,3,0] = 12*eta[i]*z_i[i]*n*li*li*li*t*alpha
        a[i,3,1] = -4*eta[i]*z_i[i]*(2*n*(n+1) - 1)*li*li*li*t*alpha
        a[i,3,2] = -2*n*li
        a[i,3,3] = -6*li

# numba speeds up the filling of the propagator matrix dramatically.
if numba_load: 
    _matFill = jit(void(float64[:,:,:], int64, float64[:], float64[:],
            float64[:], float64, float64, float64, float64), nopython=True)(_matFill)
    _matFillscale = jit(void(float64[:,:,:], int64, float64[:], float64[:], 
            float64[:], float64, float64, float64, float64), nopython=True)(_matFillscale)
    _matFilllog = jit(void(float64[:,:,:], int64, float64[:], float64[:], 
            float64[:], float64, float64, float64, float64), nopython=True)(_matFilllog)
    _matFilllogscale = jit(void(float64[:,:,:], int64, float64[:], float64[:], 
            float64[:], float64, float64, float64, float64), nopython=True)(_matFilllogscale)


def gen_viscb(n, yE, hV, params, zarray, Q=1):
    assert params.normmode == 'love', 'Must normalize parameters'
    
    # Check for individual z call
    zarray = np.asarray(zarray)
   
    parvals = params.getParams(np.r_[params.rCore, zarray, 1.])

    rho = parvals['den'][1:-1]
    g = parvals['grav'][1:-1]
    nonad = parvals['nonad'][1:-1] 
    eta = parvals['visc'][1:-1]



    rhoC = parvals['den'][0]
    gC = parvals['grav'][0]
    denC = params.denCore
    rhoS = parvals['den'][-1]

    li = 1./(2.*n+1.)

    z_i = 1./zarray

    b = np.zeros((len(zarray)+2, 4))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*hV[0] + denC*yE[4,0] +
                0.33*params.rCore*denC**2*yE[0,0]
                -rhoC*(denC-rhoC)*hV[0]*li)*li
    b[0,3] = 0.

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS*li*hV[-1]
    b[-1,3] = 0.

    # Interior points
    for i, bi in enumerate(b[1:-1]):
        hvi = 0.5*(hV[i] + hV[i+1])
        hi = 0.5*(yE[0,i] + yE[0,i+1])
        Li = 0.5*(yE[1,i] + yE[1,i+1])
        ki = 0.5*(yE[4,i] + yE[4,i+1])
        qi = 0.5*(yE[5,i] + yE[5,i+1])

        bi[0] = 0.
        bi[1] = 0.
        if Q == 1:
            bi[2] = (rho[i]*(qi +
                    (rho[i] - 4*g[i]*z_i[i])*li*hi
                    + g[i]*z_i[i]*(n+1.)*li*Li)
                    - g[i]*nonad[i]*hvi*li)
        else:
            bi[2] = (rho[i]*(qi +
                    - 4*g[i]*z_i[i]*li*hi
                    + g[i]*z_i[i]*(n+1.)*li*Li
                    + z_i[i]*(n+1.)*li*ki)
                    - g[i]*nonad[i]*hvi*li)
        bi[3] = rho[i]*(g[i]*hi + ki)*n*li*z_i[i]

    return b

class SphericalViscSMat(object):
    """Class that provides smatrix to Solvde for viscous solutions.
    
    The boundary conditions in smatrix have assumed that the equations of
    motion in propMatVisc have been nondimensionalized.
    """
    def __init__(self, n, z, params, Q=1, b=None, scaled=False, logtime=False):
        self.n = n
        self.mpt = len(z)
        self.z = z
        self.scaled = scaled
        self.logtime = logtime

        # Make sure parameters are normalized properly.
        params.normalize('love')
        self.params = params
        self.Q = Q
        self.b = b
        self.load = 1./self.params.getLithFilter(n=n) 
      
        self.updateProps(self.n, self.z, self.b)
        
    def updateProps(self, n=None, z=None, b=None, t=None):
        self.n = n or self.n
        if not self.scaled:
            self.z = self.z if z is None else z

        # Only recompute A matrix if n or z are changed.
        if n is not None or z is not None or t is not None:
            t = t or 1
            self.A = propMatVisc(self.zmids, self.n, self.params, t, self.Q, 
                                    self.scaled, self.logtime)

            if self.scaled:
                self.A = 1./self.zetamids[:,None,None]*self.A
            self.load = 1./self.params.getLithFilter(n=self.n)

        if b is not None:
            if self.scaled:
                b[1:-1] *= 1./self.zetamids[:,None]/(self.n+0.5)
            self.b = b

    @property
    def zeta(self):
        return np.arange(self.mpt)/(self.mpt-1)*(1-self.zeta_c)+self.zeta_c

    @property
    def zetamids(self):
        """Evenly spaced zetas between zeta_c and 1"""
        return (np.arange(1,self.mpt)-0.5)/(self.mpt-1)*(1-self.zeta_c)+self.zeta_c
    
    @property
    def zeta_c(self):
        return np.exp((self.n + 0.5)*(self.params.rCore - 1))

    @property
    def zetasep(self):
        return (1-self.zeta_c)/(self.mpt-1)

    @property
    def zmids(self):
        if self.scaled:
            return self.zeta2z(self.zetamids)
        else:
            return 0.5*(self.z[1:]+self.z[:-1])

    def sep(self, k):
        if self.scaled:
            return self.zetasep
        else:
            return (self.z[k] - self.z[k-1])
    def zeta2z(self, zeta):
        return 1 + np.log(zeta)/(self.n+0.5)

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        Q = self.Q
        if k == k1:      # Core-Mantle boundary conditions.            
            # Radial stress on the core.
            s[2, 4+indexv[0]] = 0.
            s[2, 4+indexv[1]] = 0.
            s[2, 4+indexv[2]] = 1.
            s[2, 4+indexv[3]] = 0.
            s[2, jsf] = y[2,0] 
                                            
            # Poloidal stress on the core.
            s[3, 4+indexv[0]] = 0.
            s[3, 4+indexv[1]] = 0.
            s[3, 4+indexv[2]] = 0.
            s[3, 4+indexv[3]] = 1.
            s[3, jsf] = y[3,0]
            
            if self.b is not None:
                s[[2,3],jsf] -= self.b[0,[2,3]]

        elif k >= k2:     # Surface boundary conditions.

            # Radial stress on surface.
            s[0, 4+indexv[0]] = 0.
            s[0, 4+indexv[1]] = 0.
            s[0, 4+indexv[2]] = 1.
            s[0, 4+indexv[3]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + self.load)

            # Poloidal stress on surface.
            s[1, 4+indexv[0]] = 0.
            s[1, 4+indexv[1]] = 0.
            s[1, 4+indexv[2]] = 0.
            s[1, 4+indexv[3]] = 1.
            s[1, jsf] = y[3, self.mpt-1]

            if self.b is not None:
                s[[0,1],jsf] -= self.b[-1, [2,3]] 

        else:           # Finite differences.
            A = 0.5*self.sep(k)*self.A[k-1]
            if self.b is None:
                b = np.zeros_like(self.z)
            else:
                b = self.sep(k)*self.b[k+1]
            interior_smatrix_fast(4, k, jsf, A, b, y, indexv, s) 
                      
        return s
    
    def checkbc(self, y, indexv):
        """Check the error at the boundary conditions for a solution array y.
        """
        s = np.zeros((4, 9))
        bot = self.smatrix(0,0,1,8,0,0,indexv, s, y)[[2,3],8]
        top = self.smatrix(1,0,1,8,0,0,indexv, s, y)[[0,1],8]
        return bot, top
