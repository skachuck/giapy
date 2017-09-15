"""
elasticlove.py
Author: Samuel B. Kachuck
Date: April 1, 2017

    Compute Elastic Love Numbers.

    Methods
    -------
    compute_love_numbers
    exp_pt_density
    propMatElas
    gen_elas_b

    Classes
    -------
    SphericalElasSMat

    Note on gravity perturbation. This code supports two definitions of the
    gravity perturbation, using the keyword Q. Q=1 is simply the radial
    derivative of the perturbation of the gravtiational potential. Q=2
    corresponds to the generalized flux, which is defined 
    $$Q_2=4\pi G U_L+\frac{\ell+1}{r}\Psi+\frac{\partial \Psi}{\partial r}.$$

"""

from __future__ import division
import numpy as np
import sys
from giapy.earth_tools.earthParams import EarthParams
# Check for numba, use if present otherwise, skip.
try:
    from giapy.numTools.solvdeJit import interior_smatrix_fast, solvde
    from numba import jit, void, int64, float64
    numba_load = True
except ImportError:
    from giapy.numTools.solvde import interior_smatrix_fast, solvde
    numba_load = False

def compute_love_numbers(ns, zarrayorgen, params, err=1e-14, Q=2, it_counts=False,
                             zgen=False, args=[], kwargs={}):
    """Compute surface elastic load love numbers for harmonic order numbers ns.

    Parameters
    ----------
    ns : array of order numbers
    zarrayorgen : array or funtion that makes array (see zgen)
    params : <giapy.earth_tools.earthParams.EarthParams> object 
        for interpolation of earth parameters to relevant points.
    err : float, maximum tolerated error for relaxation method.

    it_counts : Boolean
        Whether counts of relaxation iterations are output (default False)
    zgen : Boolean
        Is zarrayorgen a function to make zarrays (True) or an array (False)
        (default False)
    args, kwargs: optional arguments for the zaarrayorgen function.

    Returns
    -------
    hLk : (len(ns), 3) array of h, L, and k_d. 
        To get the total gravitational love number, k=n*(1+k_d).
    its : len(ns) array of iteration numbers for relaxation method 
        (if it_counts=True).
    """

    hLk = []
    if it_counts:
        its = []
    if zgen:
        zarray = zarrayorgen(ns[0], *args, **kwargs)
    else:
        zarray = zarrayorgen

    # Setup for relaxation method
    scalvElas = np.array([1., 1., 1., 1., 1., 1.])
    indexv = np.array([3,4,0,1,5,2])
    slowc = 1

    # Initial guess - subsequent orders use previous solution.
    y0 = (scalvElas*np.ones((6, len(zarray))).T).T
    
    # Main order number loop.
    #TODO add adaptive n stepsize and interpolate to interior orders.
    for n in ns:
        sys.stdout.write('Computing love number {}\r'.format(n))
        if zgen: 
            zarray = zarrayorgen(n, *args, **kwargs)
            # If not first order num, update relaxation object...
            try:
                difeqElas.updateProps(n=n, z=zarray)
            # ... otherwise create it.
            except:
                difeqElas = SphericalElasSMat(ns[0], zarray, params, Q=2)
        else:
            # If not first order num, update relaxation object...
            try:
                difeqElas.updateProps(n=n)
            # ... otherwise create it.
            except:
                difeqElas = SphericalElasSMat(ns[0], zarray, params, Q=Q)

        # Perform the relaxation for the order number and store results.
        y0, it = solvde(500, err, slowc, scalvElas, indexv, 3,
                                y0, difeqElas, False, it_count=True)
        hLk.append(y0[[0,1,4], -1])
        if it_counts:
            its.append(it)
    sys.stdout.write('\n')

    hLk=np.array(hLk).T

    # Correct n=1 case
    if ns[0] == 1:
        hLk[:2,0] += (1+hLk[2,0])
        hLk[2,0] -= (1+hLk[2,0])

    if it_counts:
        its=np.array(its)

    if it_counts:
        return hLk, its
    else:
        return hLk

def exp_pt_density(nz, delta=1., x0=0., x1=1., normed_delta=True):
    """Generate a point distribution with exponential spacing.

    Parameters
    ----------
    nz : integer - number of points to generate
    delta : float
        The exponential rate of increase (decrease if negative) of point 
        density. It is assumed to be normalized to the distance x1-x0 unless
        normed_delta=False. (Default 1.)
    x0, x1 : the range over which to distribute points (Defaults 0 to 1)
    """
    qs = np.arange(nz, dtype=float)
    if normed_delta:
        xs = (x1-x0)*delta * np.log(qs/(nz-1)*(np.exp(1./delta) - 1) + 1) + x0
    else:
        xs = x1* np.log(qs/(nz-1)*(np.exp((x1-x0)/delta) - 1) + 1) + x0
    return xs

def propMatElas(zarray, n, params, Q=2):
    """Generate the propagator matrix at all points in zarray.

    Parameters
    ----------
    zarray : array of radius values (can be singleton, returns one 6x6 array)
    n : int. The order number
    params : <giapy.earth_tools.earthParams.EarthParams> object 
        for interpolation of earth parameters to relevant points.
    Q : 1 or 2
        Flag for the definition of the gravity perturbation (see note at top of
        module).

    Returns
    -------
    a : (len(zarray), 6, 6) or (6,6). The array (or single) propagator matrix
    
    """
    assert params.normmode == 'love', 'Must normalize parameters'

    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        zarray[zarray>1] = 1
        singz = True
            
    # Interpolate the material parameters to solution points zarray.
    parvals = params.getParams(zarray)
    lam = parvals['bulk']
    mu = parvals['shear']
    rho = parvals['den']
    g = parvals['grav']
    grad_rho = np.gradient(parvals['den'])/np.gradient(zarray)

    # Common values
    beta_i = 1./(lam+2*mu)
    gamma = mu*(3*lam+2*mu)*beta_i
    z_i = 1./zarray
    l = (2.*n+1.)
    li = 1./l

    a = np.zeros((len(zarray), 6, 6))

    # Fill the matrix. It's a long for loop that can be accelerated with numba.
    _matFill(a, n, zarray, lam, mu, rho, grad_rho, g, beta_i, gamma, z_i, l, li, Q)
    
    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T


def _matFill(a, n, zarray, lam, mu, rho, grad_rho, g, beta_i, gamma, z_i, l, li, Q):
    """Fill the propagator matrix a, used internally by propMatElas."""
    for i in range(len(zarray)):
        
        # r dh/dr
        a[i,0,0] = -2*lam[i]*beta_i[i]
        a[i,0,1] = lam[i]*beta_i[i]*(n+1)
        a[i,0,2] = beta_i[i]*zarray[i]*l
        
        # r dL/dr
        a[i,1,0] = -n
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = zarray[i]/mu[i]*l
        
        # r df_L/dr
        if Q == 1:
            a[i,2,0] = (4*gamma[i]*z_i[i] - 4*rho[i]*g[i] 
                                + (rho[i]**2)*zarray[i])*li
        else:
            a[i,2,0] = 4*(gamma[i]*z_i[i] - rho[i]*g[i])*li
        a[i,2,1] = -(2*gamma[i]*z_i[i] - rho[i]*g[i])*(n+1)*li
        a[i,2,2] = -4*mu[i]*beta_i[i]
        a[i,2,3] = (n+1.)
        if Q == 2:
            a[i,2,4] = -rho[i]*(n+1)*li
        a[i,2,5] = zarray[i]*rho[i]
        
        # r dF_M/dr
        a[i,3,0] = (rho[i]*g[i]-2*gamma[i]*z_i[i])*n*li
        a[i,3,1] = 2*mu[i]*z_i[i]*(2*n*(n+1)*(lam[i]+mu[i])*beta_i[i] - 1)*li
        a[i,3,2] = -lam[i]*beta_i[i]*n
        a[i,3,3] = -3
        a[i,3,4] = rho[i]*n*li
        
        # r dk_d/dr
        if Q == 2:
            a[i,4,0] = -rho[i]*zarray[i]
        a[i,4,1] = 0.
        a[i,4,2] = 0.
        a[i,4,3] = 0.
        if Q == 2:
            a[i,4,4] = -(n+1)
        a[i,4,5] = zarray[i]*l
        
        # r dq/dr
        if Q == 1:
            a[i,5,0] = -(grad_rho[i]*zarray[i] 
                            + 4*mu[i]*rho[i]*beta_i[i])*li
            a[i,5,1] = 2*mu[i]*rho[i]*beta_i[i]*(n+1)*li
            a[i,5,2] = -rho[i]*zarray[i]*beta_i[i]
            a[i,5,3] = 0.
            a[i,5,4] = z_i[i]*n*(n+1.)*li
            a[i,5,5] = -2
        else:
            a[i,5,0] = -rho[i]*(n+1)*li
            a[i,5,1] = rho[i]*(n+1)*li
            a[i,5,2] = 0
            a[i,5,3] = 0.
            a[i,5,4] = 0
            a[i,5,5] = n-1.

# numba speeds up the filling of the propagator matrix dramatically.
if numba_load: 
    _matFill = jit(void(float64[:,:,:], int64, float64[:], float64[:], float64[:], float64[:],
        float64[:], float64[:], float64[:], float64[:], float64[:], float64,
        float64, int64), nopython=True)(_matFill)

def gen_elasb(n, hV, params, zarray, Q=1):
    """Generate viscous gravitational source terms for elastic eqs.

    Parameters
    ----------
    n : int. Order number of computation.
    hV : array of viscous vertical deformation mantle love numbers.
    params : <giapy.earth_tools.earthParams.EarthParams> object 
        for interpolation of earth parameters to relevant points.
    zarray : array of radius values (can be singleton, returns (3, 6) array)
    Q : 1 or 2
        Flag for the definition of the gravity perturbation (see note at top of
        module).

    Returns
    -------
    b - (len(zarray) + 2, 6) array
        The array of inhomogeneities (gravity sources from viscous deformation)
        at the two boundaries, and the interior mantle points.
    """
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        zarray[zarray>1] = 1
        singz = True
    assert params.normmode == 'love', 'Must normalize parameters' 
     
    parvals = params.getParams(np.r_[params.rCore, zarray, 1.])
    # CMB values (mantle side)
    rhoC = parvals['den'][0]
    gC = parvals['grav'][0]
    # CMB values (core side)
    denC = params.denCore

    # Mantle values.
    rho = parvals['den'][1:-1]
    g = parvals['grav'][1:-1]
    nonad = parvals['nonad'][1:-1]
    eta = parvals['visc'][1:-1]

    # Surface values.
    rhoS = paramSurf['den'][-1]

    z_i = 1./zarray
    l = (2.*n+1.)
    li = 1./l

    b = np.zeros((len(zarray)+2, 6))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*hV[0]*li
                -rhoC*(denC-rhoC)*hV[0]*li**2)
    b[0,3] = 0.
    b[0,4] = -(denC-rhoC)*hV[0]/(2.*n+1)
    b[0,5] = -n/params.rCore*(denC-rhoC)*hV[0]*li**2

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS*li*hV[-1]
    b[-1,3] = 0.
    b[-1,4] = 0.
    if Q == 1:
        b[-1,5] = -rhoS*li*hV[-1]
    else:
        b[-1,5] = 0.

    # Interior points
    for i, bi in enumerate(b[1:-1]):
        hvi = 0.5*(hV[i] + hV[i+1]) 

        bi[0] = 0.
        bi[1] = 0.
        bi[2] = -g[i]*nonad[i]*hvi
        bi[3] = 0.
        bi[4] = zarray[i]*nonad[i]*hvi*li
        if Q == 1:
            bi[5] = -(n+1.)*nonad[i]*hvi
        else:
            bi[5] = 0.

    return b


class SphericalElasSMat(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, params, Q=1, b=None):
        self.n = n
        self.z = z
        # Make sure parameters are normalized properly.
        params.normalize('love')
        self.params = params
        self.Q = Q
        self.b = b

        self.mpt = len(self.z)
        self.updateProps(self.n, self.z, self.b)
        
    def updateProps(self, n=None, z=None, b=None):
        self.n = n or self.n
        self.z = self.z if z is None else z
        # Only recompute A matrix if n or z are changed.
        if n is not None or z is not None:
            zmids = 0.5*(self.z[1:]+self.z[:-1])
            self.A = propMatElas(zmids, self.n, self.params, self.Q)
        if b is not None:
            self.b = b

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y): 
        Q = self.Q

        l = 2.*self.n+1.
        li = 1./l
        
        if k == k1:      # Core-Mantle boundary conditions.
                
            rCore = self.params.rCore
            paramsCore = self.params(rCore)
            gCore = paramsCore['grav']
            
            denCore = self.params.denCore
            difden = (self.params.denCore - paramsCore['den'])
            
            # Radial stress on the core.
            s[3, 6+indexv[0]] = -0.33*rCore*denCore**2*li
            s[3, 6+indexv[1]] = 0.
            s[3, 6+indexv[2]] = 1.
            s[3, 6+indexv[3]] = 0.
            s[3, 6+indexv[4]] = -denCore*li
            s[3, 6+indexv[5]] = 0.
            s[3, jsf] = (y[2,0] - 0.33*rCore*denCore**2*li*y[0,0] - 
                            denCore*li*y[4,0])
                                            
            # Poloidal stress on the core.
            s[4, 6+indexv[0]] = 0.          
            s[4, 6+indexv[1]] = 0.          
            s[4, 6+indexv[2]] = 0.          
            s[4, 6+indexv[3]] = 1.          
            s[4, 6+indexv[4]] = 0.          
            s[4, 6+indexv[5]] = 0.
            s[4, jsf] = y[3,0]

            # gravitational potential perturbation on core.
            if Q == 1:
                s[5, 6+indexv[0]] = -difden*li
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0.
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -self.n*li/rCore
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - difden*li*y[0,0]
                                    - self.n*li/rCore*y[4,0])
            else:
                s[5, 6+indexv[0]] = -denCore*li
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0.
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -1/rCore
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - denCore*li*y[0,0] - 1/rCore*y[4,0])
            if self.b is not None: 
                s[[3,4,5],jsf] -= self.b[0, [2,3,5]]
 
                

        elif k >= k2:     # Surface boundary conditions.   
            paramsSurf = self.params(1.) 
            rhoSurf = paramsSurf['den']

            # Radial stress on surface.
            s[0, 6+indexv[0]] = 0.
            s[0, 6+indexv[1]] = 0.
            s[0, 6+indexv[2]] = 1.
            s[0, 6+indexv[3]] = 0.
            s[0, 6+indexv[4]] = 0.
            s[0, 6+indexv[5]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + 1.)

            # Poloidal stress on surface.
            s[1, 6+indexv[0]] = 0.
            s[1, 6+indexv[1]] = 0.
            s[1, 6+indexv[2]] = 0.    
            s[1, 6+indexv[3]] = 1.    
            s[1, 6+indexv[4]] = 0.    
            s[1, 6+indexv[5]] = 0.    
            s[1, jsf] = y[3, self.mpt-1]
                                      
            # gravitational acceleration perturbation on surface.
            if Q == 1:
                s[2, 6+indexv[0]] = rhoSurf*li
            else:
                s[2, 6+indexv[0]] = 0.
            s[2, 6+indexv[1]] = 0.    
            s[2, 6+indexv[2]] = 0.    
            s[2, 6+indexv[3]] = 0.
            if Q == 1:
                s[2, 6+indexv[4]] = (self.n+1.)*li
            else:
                s[2, 6+indexv[4]] = 0.
            s[2, 6+indexv[5]] = 1.
            if Q == 1:
                s[2, jsf] = (y[5, self.mpt-1] + 1 
                                + rhoSurf*li*y[0,self.mpt-1]
                                + (self.n+1.)*li*y[4,self.mpt-1])
            else:
                s[2, jsf] = y[5, self.mpt-1] + 1

            if self.b is not None:
                s[[0,1,2],jsf] -= self.b[-1, [2,3,5]]
                

        else:           # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            if self.b is None:
                b = np.zeros_like(self.z)
            else:
                b = zsep*self.b[k+1]
            interior_smatrix_fast(6, k, jsf, A, b, y, indexv, s) 
                      
        return s

    def checkbc(self, y, indexv):
        """Check the error at the boundary conditions for a solution array y.
        """
        s = np.zeros((6, 13))
        bot = self.smatrix(0,0,1,12,0,0,indexv, s, y)[[3,4,5],12]
        top = self.smatrix(1,0,1,12,0,0,indexv, s, y)[[0,1,2],12]
        return bot, top
