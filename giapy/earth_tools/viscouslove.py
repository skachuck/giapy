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

import numpy as np

def propMatVisc(zarray, n, params, Q=1):
    """Generate the propagator matrix at all points in zarray. Should have
    shape (len(zarray), 6, 6)
    
    """
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        singz = True
        
        
    r = params.norms['r']
    muStar = params.norms['mu']
    G = params.G
    
    parvals = params.getParams(zarray/r)
    rho = parvals['den']
    g = parvals['grav']
    grad_rho = np.gradient(parvals['den'])/np.gradient(zarray)#parvals['dend']/r
    #grad_rho_na = parvals['nonad']
    eta = parvals['visc']*params.norms['eta']


    z_i = 1./zarray
    
    delsq = -n*(n+1)

    a = np.zeros((len(zarray), 4, 4))

    for i in range(len(zarray)):
        
        # r dU_L/dr
        a[i,0,0] = -2
        a[i,0,1] = -delsq
        a[i,0,2] = 0
        
        # r dU_M/dr
        a[i,1,0] = -1.
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = zarray[i]/eta[i]
        
        # r dT_L/dr
        a[i,2,0] = 12*eta[i]*z_i[i]
        a[i,2,1] = 6*eta[i]*z_i[i]*delsq
        a[i,2,2] = 0.
        a[i,2,3] = -delsq
        
        # r dT_M/dr
        a[i,3,0] = -6*eta[i]*z_i[i]
        a[i,3,1] = -2*eta[i]*(2*delsq + 1)*z_i[i]
        a[i,3,2] = -1.
        a[i,3,3] = -3.

    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T

def gen_viscb(n, yE, uV, params, zarray, Q=1):
    # Check for individual z call
    zarray = np.asarray(zarray)
        
        
    re = params.norms['r']
    G = params.G
    
    parvals = params.getParams(zarray/re)

    rho = parvals['den']
    g = parvals['grav']
    
    paramSurf = params.getParams(1.)
    g0 = paramSurf['grav']
    re = params.norms['r']
    rhobar = g0/params.G/re
    
    g = g
    rho = rho
    eta = parvals['visc']*params.norms['eta']

    paramCore = params.getParams(params.rCore)
    rhoC = paramCore['den']
    gC = paramCore['grav']
    denC = params.denCore
    
    rhoS = paramSurf['den']

    z_i = 1./zarray

    b = np.zeros((len(zarray)+2, 4))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*uV[0] + denC*yE[4,0] + 0.33*G*re*params.rCore*denC**2*yE[0,0])
    #b[0,2] = ((denC-rhoC)*gC*uV[0] + denC*yE[4,0] + gC*denC*yE[0,0])
    b[0,3] = 0.

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS*uV[-1]
    b[-1,3] = 0.

    # Interior points
    for i, bi in enumerate(b[1:-1]):
        Qi = 0.5*(yE[5,i] + yE[5,i+1])
        UlVi = 0.5*(uV[i] + uV[i+1])
        UlEi = 0.5*(yE[0,i] + yE[0,i+1])
        UmEi = 0.5*(yE[1,i] + yE[1,i+1])
        Psii = 0.5*(yE[4,i] + yE[4,i+1])

        bi[0] = 0.
        bi[1] = 0.
        if Q == 1:
            bi[2] = rho[i]*(Qi - g[i]*UlVi +
                        (G*rho[i] - 4*g[i]*z_i[i])*UlEi +
                        g[i]*z_i[i]*n*(n+1)*UmEi)
        else:
            bi[2] = rho[i]*(Qi - 4*g[i]*z_i[i]*UlEi + 
                            g[i]*n*(n+1)*z_i[i]*UmEi -
                            (n+1)*z_i[i]*Psii)
        bi[3] = (rho[i]*Psii + rho[i]*g[i]*UlEi)*z_i[i]

    return b

class SphericalViscSMat(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, params, Q=1, b=None):
        self.n = n
        self.z = z
        self.params = params
        self.Q = Q
        self.b = b

        self.mpt = len(self.z)
        self.updateProps()
        
    def updateProps(self, n=None, z=None, b=None):
        self.n = n or self.n
        self.z = self.z if z is None else z
        zmids = 0.5*(self.z[1:]+self.z[:-1])
        self.A = propMatVisc(zmids, self.n, self.params, self.Q)

        self.b = b or self.b

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        Q = self.Q
        if k == k1:      # Core-Mantle boundary conditions.
            rCore = self.params.rCore
            paramsCore = self.params(rCore)
            rhoCore = paramsCore['den']
            gCore = paramsCore['grav']
            
            denCore = self.params.denCore
            difden = denCore-rhoCore
            G = self.params.G

            rstar = self.params.norms['r']
            mustar = self.params.norms['mu']
            
            # Radial stress on the core.
            s[2, 4+indexv[0]] = 0.
            s[2, 4+indexv[1]] = 0.
            s[2, 4+indexv[2]] = 1.
            s[2, 4+indexv[3]] = 0.
            s[2, jsf] = y[2,0]#(y[2,0] - 0.33*G*rstar*rCore*denCore**2*y[0,0] - 
                            #denCore*y[4,0])
                                            
            # Poloidal stress on the core.
            s[3, 4+indexv[0]] = 0.          
            s[3, 4+indexv[1]] = 0.          
            s[3, 4+indexv[2]] = 0.          
            s[3, 4+indexv[3]] = 1.                  
            s[3, jsf] = y[3,0]

            if self.b is not None:
                s[:,jsf] -= self.b[0]
                

        elif k >= k2:     # Surface boundary conditions.
            paramsSurf = self.params(1.)
            gSurf = paramsSurf['grav']
            rhoSurf = paramsSurf['den']
            G = self.params.G
            

            rstar = self.params.norms['r']
            mustar = self.params.norms['mu']

            # Radial stress on surface.
            s[0, 4+indexv[0]] = 0.
            s[0, 4+indexv[1]] = 0.
            s[0, 4+indexv[2]] = 1.
            s[0, 4+indexv[3]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + gSurf)

            # Poloidal stress on surface.
            s[1, 4+indexv[0]] = 0.
            s[1, 4+indexv[1]] = 0.
            s[1, 4+indexv[2]] = 0.    
            s[1, 4+indexv[3]] = 1.      
            s[1, jsf] = y[3, self.mpt-1]
            
            if self.b is not None:
                s[:, jsf] -= self.b[-1]

        else:           # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            if self.b is None:
                b = np.zeros_like(self.z)
            else:
                b = zsep*self.b[k]
            interior_smatrix_fast(4, k, jsf, A, b, y, indexv, s) 
                      
        return s


def propMatVisc_norm(zarray, n, params, Q=1):
    """Generate the viscous love-number propagator matrix for zarray. 
    
    Inhomogeneities from elastic deformation and viscous migration of 
    nonadiabatic density gradients are added using gen_viscb_norm.

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
        or (6,6).
    """
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

def gen_viscb_norm(n, yE, uV, params, zarray, Q=1):
    # Check for individual z call
    zarray = np.asarray(zarray)
 
    re = params.norms['r']
    G = params.G
    
    parvals = params.getParams(zarray)

    rho = parvals['den']
    g = parvals['grav']
    nonad = parvals['nonad']
    
    paramSurf = params.getParams(1.)
    g0 = paramSurf['grav']
    re = params.norms['r']
    rhobar = g0/params.G/re
    
    g = g/g0
    rho = rho/rhobar
    eta = parvals['visc']

    paramCore = params.getParams(params.rCore)
    rhoC = paramCore['den']/rhobar
    gC = paramCore['grav']/g0
    denC = params.denCore/rhobar
    rhoS = paramSurf['den']/rhobar
    nonad = nonad/rhobar

    z_i = 1./zarray

    b = np.zeros((len(zarray)+2, 4))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*uV[0] + denC*yE[4,0] +
                0.33*params.rCore*denC**2*yE[0,0]
                -rhoC*(denC-rhoC)*uV[0]/(2.*n+1.))/(2.*n+1.)
    b[0,3] = 0.

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS/(2.*n+1.)*uV[-1]
    b[-1,3] = 0.

    # Interior points
    for i, bi in enumerate(b[1:-1]):
        hvi = 0.5*(uV[i] + uV[i+1])
        hi = 0.5*(yE[0,i] + yE[0,i+1])
        Li = 0.5*(yE[1,i] + yE[1,i+1])
        ki = 0.5*(yE[4,i] + yE[4,i+1])
        qi = 0.5*(yE[5,i] + yE[5,i+1])

        bi[0] = 0.
        bi[1] = 0.
        if Q == 1:
            bi[2] = (rho[i]*(qi +
                    (rho[i] - 4*g[i]*z_i[i])/(2*n+1)*hi +
                    g[i]*z_i[i]*(n+1.)/(2.*n+1.)*Li)
                    + g[i]*nonad[i]*hvi)
        else:
            bi[2] = rho[i]*(qi - g[i]*hvi +
                    - 4*g[i]*z_i[i]/(2.*n+1.)*hi
                    + g[i]*z_i[i]*(n+1.)/(2.*n+1.)*Li
                    + z_i*(n+1.)/(2.*n+1.)*ki)
        bi[3] = rho[i]*(g[i]*hi + ki)*n/(2.*n+1.)*z_i[i]

    return b

class SphericalViscSMat_norm(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, params, Q=1, b=None):
        self.n = n
        self.z = z
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
            self.A = propMatVisc_norm(zmids, self.n, self.params, self.Q)
        if b is not None:
            self.b = b

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        Q = self.Q
        if k == k1:      # Core-Mantle boundary conditions.
            rCore = self.params.rCore
            paramsCore = self.params(rCore)
            rhoCore = paramsCore['den']
            gCore = paramsCore['grav']
            
            denCore = self.params.denCore
            difden = denCore-rhoCore
            G = self.params.G

            rstar = self.params.norms['r']
            mustar = self.params.norms['mu']
            
            # Radial stress on the core.
            s[2, 4+indexv[0]] = 0.
            s[2, 4+indexv[1]] = 0.
            s[2, 4+indexv[2]] = 1.
            s[2, 4+indexv[3]] = 0.
            s[2, jsf] = y[2,0]#(y[2,0] - 0.33*G*rstar*rCore*denCore**2*y[0,0] - 
                            #denCore*y[4,0])
                                            
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
            s[0, jsf] = (y[2, self.mpt-1] + 1)

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

def interior_smatrix_fast(n, k, jsf, A, b, y, indexv, s):
    for i in range(n):
        rgt = 0.
        for j in range(n):
            if i==j:
                s[i, indexv[j]]   = -1. - A[i,j]
                s[i, n+indexv[j]] =  1. - A[i,j]
            else:
                s[i, indexv[j]]   = -A[i,j]
                s[i, n+indexv[j]] = -A[i,j]
            rgt += A[i,j] * (y[j, k] + y[j, k-1])
        s[i, jsf] = y[i, k] - y[i, k-1] - rgt - b[i]
