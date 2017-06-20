"""
elasticlove.py

    Compute Elastic Love Numbers 
"""

import numpy as np
from giapy.earth.earthParams import EarthParams
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
            a[i,2,0] = 4*(gamma[i]*z_i[i] - rho[i]*g[i]) + G*rho[i]**2
        else:
            a[i,2,0] = 4*(gamma[i]*z_i[i] - rho[i]*g[i])
        a[i,2,1] = (2*gamma[i]*z_i[i]-rho[i]*g[i])*delsq
        a[i,2,2] = -4*mu[i]*beta_i[i]
        a[i,2,3] = -delsq
        if Q == 1:
            a[i,2,4] = 0
        else:
            a[i,2,4] = -(n+1)*rho[i]
        a[i,2,5] = zarray[i]*rho[i]
        
        # r dT_M/dr
        a[i,3,0] = -2*gamma[i]*z_i[i]+rho[i]*g[i]
        a[i,3,1] = -2*mu[i]*(2*delsq*(lam[i]+mu[i])*beta_i[i] + 1)*z_i[i]
        a[i,3,2] = -lam[i]*beta_i[i]
        a[i,3,3] = -3
        a[i,3,4] = rho[i]
        
        # r dPsi/dr 
        if Q == 1:
            a[i,4,0] = 0.
        else: 
            a[i,4,0] = -G*rho[i]*zarray[i]
        a[i,4,1] = 0.
        a[i,4,2] = 0.
        if Q == 1:
            a[i,4,3] = 0.
        else:
            a[i,4,3] = -(n+1)
        a[i,4,4] = 0.
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
        if n is None: n = self.n
        self.n = n
        if z is None: z = self.z
        self.z = z
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
            if Q == 1:
                s[3, 6+indexv[0]] = -gCore*denCore
            else:
                s[3, 6+indexv[0]] = -0.33*G*rstar*rCore*denCore**2
            s[3, 6+indexv[1]] = 0.
            s[3, 6+indexv[2]] = 1.
            s[3, 6+indexv[3]] = 0.
            s[3, 6+indexv[4]] = -denCore    
            s[3, 6+indexv[5]] = 0.
            if Q == 1:
                s[3, jsf] = (y[2,0] - gCore*denCore*y[0,0] - 
                            denCore*y[4,0])
            else:
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
                s[5, jsf] = (y[5,0] - G*difden*y[0,0]-(self.n)/(rCore*rstar)*y[4,0])
            else:
                s[5, 6+indexv[0]] = 0#-G*denCore
                s[5, 6+indexv[1]] = 0.
                s[5, 6+indexv[2]] = -3/rCore/rstar/rhoCore
                s[5, 6+indexv[3]] = 0
                s[5, 6+indexv[4]] = -2*(self.n-1)/rCore/rstar
                s[5, 6+indexv[5]] = 1.
                #s[5, jsf] = (y[5,0] - G*denCore*y[0,0] - 2*(self.n-1)/rCore/rstar*y[4,0])
                s[5, jsf] = (y[5,0] - 3/rCore/rstar/rhoCore*y[2,0] - 2*(self.n-1)/rCore/rstar*y[4,0])
 
                

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
                                      
            # gravitational potential perturbation on surface.
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
                s[2, jsf] = (y[5, self.mpt-1] + G +
                                 G*rhoSurf*y[0,self.mpt-1]+
                                 (self.n+1)*y[4,self.mpt-1]/rstar)
            else:
                s[2, jsf] = y[5, self.mpt-1] + G

        else:           # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            tmp = np.zeros_like(self.z)
            interior_smatrix_fast(6, k, jsf, A, tmp, y, indexv, s) 
                      
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
