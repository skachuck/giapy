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

def propMatVisc(zarray, n, params, Q=1):
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

    for i in range(len(zarray)):
        
        # r d\dt{h}/dr
        a[i,0,0] = -2
        a[i,0,1] = n+1
        a[i,0,2] = 0#(2*n+1)*zarray[i]/eta[i]
        
        # r d\dt{L}/dr
        a[i,1,0] = -n
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = 2*zarray[i]*(2*n+1)/eta[i]
        
        # r dT_L/dr 
        a[i,2,0] = 6*eta[i]*z_i[i]/(2*n+1) 
        a[i,2,1] = -3*eta[i]*z_i[i]*(n+1)/(2*n+1)
        a[i,2,2] = 0.
        a[i,2,3] = n+1
        
        # r dT_M/dr
        a[i,3,0] = -3*eta[i]*z_i[i]*n/(2*n+1)
        a[i,3,1] = eta[i]*z_i[i]*(2*n*(n+1) - 1)/(2*n+1)
        a[i,3,2] = -n
        a[i,3,3] = -3

    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T

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


    z_i = 1./zarray

    b = np.zeros((len(zarray)+2, 4))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*hV[0] + denC*yE[4,0] +
                0.33*params.rCore*denC**2*yE[0,0]
                -rhoC*(denC-rhoC)*hV[0]/(2.*n+1.))/(2.*n+1.)
    b[0,3] = 0.

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS/(2.*n+1.)*hV[-1]
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
                    (rho[i] - 4*g[i]*z_i[i])/(2*n+1)*hi
                    + g[i]*z_i[i]*(n+1.)/(2.*n+1.)*Li)
                    - g[i]*nonad[i]*hvi)
        else:
            bi[2] = (rho[i]*(qi +
                    - 4*g[i]*z_i[i]/(2.*n+1.)*hi
                    + g[i]*z_i[i]*(n+1.)/(2.*n+1.)*Li
                    + z_i[i]*(n+1.)/(2.*n+1.)*ki)
                    - g[i]*nonad[i]*hvi)
        bi[3] = rho[i]*(g[i]*hi + ki)*n/(2.*n+1.)*z_i[i]

    return b

class SphericalViscSMat(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, params, Q=1, b=None):
        self.n = n
        self.z = z
        self.params = params
        self.load = 1./self.params.getLithFilter(n=n)
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
            self.A = propMatVisc(zmids, self.n, self.params, self.Q)
            self.load = 1./self.params.getLithFilter(n=n)
        if b is not None:
            self.b = b

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
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            if self.b is None:
                b = np.zeros_like(self.z)
            else:
                b = zsep*self.b[k+1]
            interior_smatrix_fast(4, k, jsf, A, b, y, indexv, s) 
                      
        return s
