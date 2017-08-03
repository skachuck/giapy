"""
elasticlove.py
Author: Samuel B. Kachuck
Date: April 1, 2017

    Compute Elastic Love Numbers.

    Note on gravity perturbation. This code supports two definitions of the
    gravity perturbation, using the keyword Q. Q=1 is simply the radial
    derivative of the perturbation of the gravtiational potential. Q=2
    corresponds to the generalized flux, which is defined 
    $$Q_2=4\pi G U_L+\frac{\ell+1}{r}\Psi+\frac{\partial \Psi}{\partial r}.$$

"""

import numpy as np
from giapy.earth_tools.earthParams import EarthParams
from giapy.numTools.solvde import solvde

def propMatElas(zarray, n, params, Q=1):
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
    lam = muStar*parvals['bulk']
    mu = muStar*parvals['shear']
    rho = parvals['den']
    g = parvals['grav']
    grad_rho = np.gradient(parvals['den'])/np.gradient(zarray)#parvals['dend']/r
    #grad_rho_na = parvals['nonad']

    beta_i = 1./(lam+2*mu)
    gamma = mu*(3*lam+2*mu)*beta_i

    z_i = 1./zarray
    
    delsq = -n*(n+1)

    a = np.zeros((len(zarray), 6, 6))

    for i in range(len(zarray)):
        
        # r dU_L/dr
        a[i,0,0] = -2*lam[i]*beta_i[i]
        a[i,0,1] = -lam[i]*beta_i[i]*delsq
        a[i,0,2] = beta_i[i]*zarray[i]
        
        # r dU_M/dr
        a[i,1,0] = -1.
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = zarray[i]/mu[i]
        
        # r dT_L/dr
        if Q == 1:
            a[i,2,0] = (4*(gamma[i]*z_i[i] - rho[i]*g[i]) +
                            G*rho[i]**2*zarray[i])
        else:
            a[i,2,0] = 4*(gamma[i]*z_i[i] - rho[i]*g[i])
        a[i,2,1] = (2*gamma[i]*z_i[i]-rho[i]*g[i])*delsq
        a[i,2,2] = -4*mu[i]*beta_i[i]
        a[i,2,3] = -delsq
        if Q == 2: 
            a[i,2,4] = -(n+1)*rho[i]
        a[i,2,5] = zarray[i]*rho[i]
        
        # r dT_M/dr
        a[i,3,0] = -2*gamma[i]*z_i[i]+rho[i]*g[i]
        a[i,3,1] = -2*mu[i]*(2*delsq*(lam[i]+mu[i])*beta_i[i] + 1)*z_i[i]
        a[i,3,2] = -lam[i]*beta_i[i]
        a[i,3,3] = -3
        a[i,3,4] = rho[i]
        
        # r dPsi/dr 
        if Q == 2: 
            a[i,4,0] = -G*rho[i]*zarray[i]
        a[i,4,1] = 0.
        a[i,4,2] = 0.
        a[i,4,3] = 0.
        if Q == 2:
            a[i,4,4] = -(n+1)
        a[i,4,5] = zarray[i]
        
        # r dQ/dr
        if Q == 1:
            a[i,5,0] = -G*(4*rho[i]*mu[i]*beta_i[i]+zarray[i]*grad_rho[i])
            a[i,5,1] = -2*delsq*G*rho[i]*mu[i]*beta_i[i]
            a[i,5,2] = -G*zarray[i]*rho[i]*beta_i[i]
            a[i,5,3] = 0.
            a[i,5,4] = -delsq*z_i[i]
            a[i,5,5] = -2
        else:
            a[i,5,0] = -G*rho[i]*(n+1)
            a[i,5,1] = -delsq*G*rho[i]
            a[i,5,2] = 0
            a[i,5,3] = 0.
            a[i,5,4] = 0
            a[i,5,5] = n-1

    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T

class SphericalElasSMat(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, params, Q=1):
        self.n = n
        self.z = z
        self.params = params
        self.Q = Q

        self.mpt = len(self.z)
        self.updateProps()
        
    def updateProps(self, n=None, z=None):
        self.n = n or self.n
        self.z = self.z if z is None else z
        zmids = 0.5*(self.z[1:]+self.z[:-1])
        self.A = propMatElas(zmids, self.n, self.params, self.Q)

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
            s[3, 6+indexv[0]] = -0.33*G*rstar*rCore*denCore**2#-gCore*denCore
            s[3, 6+indexv[1]] = 0.
            s[3, 6+indexv[2]] = 1.
            s[3, 6+indexv[3]] = 0.
            s[3, 6+indexv[4]] = -denCore    
            s[3, 6+indexv[5]] = 0.
            s[3, jsf] = (y[2,0] - 0.33*G*rstar*rCore*denCore**2*y[0,0] -
                            denCore*y[4,0])
                                            
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
                s[5, 6+indexv[0]] = -G*difden
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0.
                s[5, 6+indexv[3]] = 0.
                s[5, 6+indexv[4]] = -(self.n)/rCore/rstar
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - G*difden*y[0,0] - self.n/rCore/rstar*y[4,0])
            else:
                s[5, 6+indexv[0]] = -G*denCore
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -(2*self.n+1)/rCore/rstar
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - G*denCore*y[0,0] - (2*self.n+1)/rCore/rstar*y[4,0]) 
 
                

        elif k >= k2:     # Surface boundary conditions.
            paramsSurf = self.params(1.)
            gSurf = paramsSurf['grav']
            rhoSurf = paramsSurf['den']
            G = self.params.G

            rstar = self.params.norms['r']
            mustar = self.params.norms['mu']

            # Radial stress on surface.
            s[0, 6+indexv[0]] = 0.
            s[0, 6+indexv[1]] = 0.
            s[0, 6+indexv[2]] = 1.
            s[0, 6+indexv[3]] = 0.
            s[0, 6+indexv[4]] = 0.
            s[0, 6+indexv[5]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + gSurf)

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
                s[2, 6+indexv[0]] = G*rhoSurf
            else:
                s[2, 6+indexv[0]] = 0.
            s[2, 6+indexv[1]] = 0.    
            s[2, 6+indexv[2]] = 0.    
            s[2, 6+indexv[3]] = 0.
            if Q == 1:
                s[2, 6+indexv[4]] = (self.n+1)/rstar
            else:
                s[2, 6+indexv[4]] = 0.
            s[2, 6+indexv[5]] = 1.
            if Q == 1:
                s[2, jsf] = (y[5, self.mpt-1] + G
                                + G*rhoSurf*y[0,self.mpt-1]
                                + (self.n+1)*y[4,self.mpt-1]/rstar)
            else:
                s[2, jsf] = y[5, self.mpt-1] + G

        else:           # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            tmp = np.zeros_like(self.z)
            interior_smatrix_fast(6, k, jsf, A, tmp, y, indexv, s) 
                      
        return s

def propMatElas_norm(zarray, n, params, Q=1):
    """Generate the propagator matrix at all points in zarray. Should have
    shape (len(zarray), 6, 6)
    
    """
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        zarray[zarray>1] = 1
        singz = True
        

    r = params.norms['r']
    muStar = params.norms['mu']
    G = params.G
    
    parvals = params.getParams(zarray)
    lam = muStar*parvals['bulk']
    mu = muStar*parvals['shear']
    rho = parvals['den']
    g = parvals['grav']

    
    g0 = params.getParams(1.)['grav']
    re = params.norms['r']
    rhobar = g0/params.G/re

    mu = mu/rhobar/re/g0
    lam = lam/rhobar/re/g0
    g = g/g0
    rho = rho/rhobar
    grad_rho = np.gradient(rho)/np.gradient(zarray)#grad_rho/rhobar
 
    #grad_rho_na = parvals['nonad']

    beta_i = 1./(lam+2*mu)
    gamma = mu*(3*lam+2*mu)*beta_i

    z_i = 1./zarray

    a = np.zeros((len(zarray), 6, 6))

    for i in range(len(zarray)):
        
        # r dh/dr
        a[i,0,0] = -2*lam[i]*beta_i[i]
        a[i,0,1] = lam[i]*beta_i[i]*(n+1)
        a[i,0,2] = beta_i[i]*zarray[i]*(2*n+1)
        
        # r dL/dr
        a[i,1,0] = -n
        a[i,1,1] = 1.
        a[i,1,2] = 0.
        a[i,1,3] = zarray[i]/mu[i]*(2*n+1)
        
        # r df_L/dr
        if Q == 1:
            a[i,2,0] = (4*gamma[i]*z_i[i] - 4*rho[i]*g[i] 
                                + (rho[i]**2)*zarray[i])/(2*n+1)
        else:
            a[i,2,0] = 4*(gamma[i]*z_i[i] - rho[i]*g[i])/(2*n+1)
        a[i,2,1] = -(2*gamma[i]*z_i[i] - rho[i]*g[i])*(n+1)/(2*n+1)
        a[i,2,2] = -4*mu[i]*beta_i[i]
        a[i,2,3] = (n+1.)
        if Q == 2:
            a[i,2,4] = -rho[i]*(n+1)/(2*n+1)
        a[i,2,5] = zarray[i]*rho[i]
        
        # r dF_M/dr
        a[i,3,0] = (rho[i]*g[i]-2*gamma[i]*z_i[i])*n/(2*n+1)
        a[i,3,1] = 2*mu[i]*z_i[i]*(2*n*(n+1)*(lam[i]+mu[i])*beta_i[i] - 1)/(2*n+1)
        a[i,3,2] = -lam[i]*beta_i[i]*n
        a[i,3,3] = -3
        a[i,3,4] = rho[i]*n/(2*n+1)
        
        # r dk_d/dr
        if Q == 2:
            a[i,4,0] = -rho[i]*zarray[i]
        a[i,4,1] = 0.
        a[i,4,2] = 0.
        a[i,4,3] = 0.
        if Q == 2:
            a[i,4,4] = -(n+1)
        a[i,4,5] = zarray[i]*(2*n+1.)
        
        # r dq/dr
        if Q == 1:
            a[i,5,0] = -(grad_rho[i]*zarray[i] 
                            + 4*mu[i]*rho[i]*beta_i[i])/(2*n+1)
            a[i,5,1] = 2*mu[i]*rho[i]*beta_i[i]*(n+1)/(2*n+1)
            a[i,5,2] = -rho[i]*zarray[i]*beta_i[i]
            a[i,5,3] = 0.
            a[i,5,4] = z_i[i]*n*(n+1.)/(2.*n+1.)
            a[i,5,5] = -2
        else:
            a[i,5,0] = -rho[i]*(n+1)/(2*n+1)
            a[i,5,1] = rho[i]*(n+1)/(2*n+1)
            a[i,5,2] = 0
            a[i,5,3] = 0.
            a[i,5,4] = 0
            a[i,5,5] = n-1.

    if singz:
        return z_i*a[0]
    else:
        return (z_i*a.T).T

def gen_elasb_norm(n, uV, params, zarray, Q=1):
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
    nonad = nonad/rhobar

    eta = parvals['visc']

    paramCore = params.getParams(params.rCore)
    rhoC = paramCore['den']/rhobar
    gC = paramCore['grav']/g0
    denC = params.denCore/rhobar
    
    rhoS = paramSurf['den']/rhobar

    z_i = 1./zarray

    b = np.zeros((len(zarray)+2, 6))

    # Lower Boundary Condition inhomogeneity
    b[0,0] = 0.
    b[0,1] = 0.
    b[0,2] = ((denC-rhoC)*gC*uV[0]/(2.*n+1.)
                -rhoC*(denC-rhoC)*uV[0]/(2.*n+1)**2)
    b[0,3] = 0.
    b[0,4] = -(denC-rhoC)*uV[0]/(2.*n+1)
    b[0,5] = -n/params.rCore*(denC-rhoC)*uV[0]/(2.*n+1)**2

    # Upper Boundary Condition inhomogeneity
    b[-1,0] = 0.
    b[-1,1] = 0.
    b[-1,2] = -rhoS/(2.*n+1.)*uV[-1]
    b[-1,3] = 0.
    b[-1,4] = 0.
    if Q == 1:
        b[-1,5] = -rhoS/(2.*n+1.)*uV[-1]
    else:
        b[-1,5] = 0.

    # Interior points
    for i, bi in enumerate(b[1:-1]):
        hvi = 0.5*(uV[i] + uV[i+1]) 

        bi[0] = 0.
        bi[1] = 0.
        bi[2] = -g[i]*nonad[i]*hvi
        bi[3] = 0.
        bi[4] = zarray[i]*nonad[i]*hvi/(2.*n+1.)
        if Q == 1:
            bi[5] = -(n+1.)*nonad[i]*hvi
        else:
            bi[5] = 0.

    return b


class SphericalElasSMat_norm(object):
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
            self.A = propMatElas_norm(zmids, self.n, self.params, self.Q)
        if b is not None:
            self.b = b

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        
        paramSurf = self.params(1.)
        rhobar = paramSurf['grav']/self.params.G/self.params.norms['r']
        Q = self.Q
        
        if k == k1:      # Core-Mantle boundary conditions.
            
            g0 = self.params.getParams(1.)['grav']
            re = self.params.norms['r']
            rhobar = g0/self.params.G/re
            
            rCore = self.params.rCore
            paramsCore = self.params(rCore)
            gCore = paramsCore['grav']/g0
            
            denCore = self.params.denCore/rhobar
            difden = (self.params.denCore - paramsCore['den'])/rhobar
            
            # Radial stress on the core.
            s[3, 6+indexv[0]] = -0.33*rCore*denCore**2/(2*self.n+1)
            s[3, 6+indexv[1]] = 0.
            s[3, 6+indexv[2]] = 1.
            s[3, 6+indexv[3]] = 0.
            s[3, 6+indexv[4]] = -denCore/(2*self.n+1)
            s[3, 6+indexv[5]] = 0.
            s[3, jsf] = (y[2,0] - 0.33*rCore*denCore**2/(2*self.n+1)*y[0,0] - 
                            denCore/(2*self.n+1)*y[4,0])
                                            
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
                s[5, 6+indexv[0]] = -difden/(2.*self.n+1.)
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0.
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -self.n/(2.*self.n+1.)/rCore
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - difden/(2.*self.n+1.)*y[0,0]
                                    - self.n/(2.*self.n+1.)/rCore*y[4,0])
            else:
                s[5, 6+indexv[0]] = -denCore/(2.*self.n+1.)
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = 0.
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -1/rCore
                s[5, 6+indexv[5]] = 1.
                s[5, jsf] = (y[5,0] - denCore/(2*self.n+1)*y[0,0] - 1/rCore*y[4,0])
            if self.b is not None: 
                s[[3,4,5],jsf] -= self.b[0, [2,3,5]]
 
                

        elif k >= k2:     # Surface boundary conditions.
            g0 = self.params.getParams(1.)['grav']
            re = self.params.norms['r']
            rhobar = g0/self.params.G/re
            
            paramsSurf = self.params(1.) 
            rhoSurf = paramsSurf['den']/rhobar

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
                s[2, 6+indexv[0]] = rhoSurf/(2.*self.n+1.)
            else:
                s[2, 6+indexv[0]] = 0.
            s[2, 6+indexv[1]] = 0.    
            s[2, 6+indexv[2]] = 0.    
            s[2, 6+indexv[3]] = 0.
            if Q == 1:
                s[2, 6+indexv[4]] = (self.n+1.)/(2.*self.n+1.)
            else:
                s[2, 6+indexv[4]] = 0.
            s[2, 6+indexv[5]] = 1.
            if Q == 1:
                s[2, jsf] = (y[5, self.mpt-1] + 1 
                                + rhoSurf/(2.*self.n+1.)*y[0,self.mpt-1]
                                + (self.n+1.)/(2.*self.n+1.)*y[4,self.mpt-1])
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
        s = np.zeros((6, 13))
        bot = self.smatrix(0,0,1,12,0,0,indexv, s, y)[[3,4,5],12]
        top = self.smatrix(1,0,1,12,0,0,indexv, s, y)[[0,1,2],12]
        return bot, top



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



def compute_love_numbers(ns, zarrayorgen, params, err=1e-14, Q=2, it_counts=False,
                            normed=False, zgen=False, args=[], kwargs={}):

    hlk = []
    if it_counts:
        its = []

    if zgen:
        zarray = zarrayorgen(ns[0], *args, **kwargs)
    else:
        zarray = zarrayorgen

    if normed:
        difeqElas = SphericalElasSMat_norm(ns[0], zarray, params, Q=2)
        scalvElas = np.array([1., 1., 1., 1., 1., 1.])
    else:
        difeqElas = SphericalElasSMat(ns[0], zarray, params, Q=2)
        scalvElas = np.array([1e-02, 1e-02, 1e+02, 1e+01, 1e+01, 1e-07])

    indexv = np.array([3,4,0,1,5,2])
    slowc = 1
    y0 = (scalvElas*np.ones((6, len(zarray))).T).T
    
    for n in ns:
        if zgen: 
            zarray = zarrayorgen(n)
            difeqElas.updateProps(n=n, z=zarray)
        else:
            difeqElas.updateProps(n=n)
        yE = solvde(500, err, slowc, scalvElas, indexv, 3,
                                y0, difeqElas, False)
        
        y0 = yE.y
        if not normed:
            scalvElas = 10**np.floor(np.log10(np.max(np.abs(y0), axis=1)))
        hlk.append(yE.y[[0,1,4], -1])
        if it_counts:
            its.append(yE.it)
    hlk=np.array(hlk).T
    if it_counts:
        its=np.array(its)
    
    if it_counts:
        return its, hlk
    else:
        return hlk

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

def nlayer_experiment(nzs, fname, params, zarraygen, args=[], kwargs={},
                        ns=range(2,1001), normed=False, err=1e-14, zgen=False):
    result = {}
    for nz in nzs:
        zarray = zarraygen(nz, *args, **kwargs)
        t0 = time.time()
        try:
            its, hlk = compute_love_numbers(ns, zarray, params, it_counts=True,
                                            normed=normed, err=err, zgen=zgen)
            result[nz] = its, hlk, time.time() - t0
        except ValueError: 
            result[nz] = 'solvde presumed failed'
            continue
        except:
            raise
    pickle.dump(result, open(fname, 'w'))

if __name__=='__main__':
    import time, pickle
    nzs = [1000, 500, 300, 100, 85, 70, 55, 40, 25]
    
    prem = EarthParams()
    rE = prem.norms['r']
    rCore = prem.rCore

    unigen = lambda nz: np.linspace(rCore, 1., nz)*rE
    #nlayer_experiment(nzs, 'nz_notnormed_uniform.p', prem, unigen)
    #nlayer_experiment(nzs, 'nz_notnormed_0p22uniexp.p', prem, exp_pt_density,
    #                   args=[0.22, rCore*rE, rE])
    unigenn = lambda nz: np.linspace(rCore, 1., nz)
    #nlayer_experiment(nzs, 'nz_normed_uniform.p', prem, unigenn, normed=True)
    #nlayer_experiment(nzs, 'nz_normed_0p22uniexp.p', prem, exp_pt_density, 
    #                    normed=True, args=[0.22, rCore, 1.])
    #nlayer_experiment(nzs, 'nz_normed_0p1uniexp.p', prem, exp_pt_density, 
    #                    normed=True, args=[0.1, rCore, 1.])
    
    def kbasegen(nz):
        def kbase(n):
            return exp_pt_density(nz, np.log(n)/(2*n+1), rCore, 1.)
        return kbase
    nlayer_experiment(nzs, 'nz_normed_lognkbase.p', prem, kbasegen,
                        normed=True, zgen=True)
    
