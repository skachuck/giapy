"""
viscellove.py
Author: Samuel B. Kachuck
Date: August 1, 2017

    Compute viscoelastic decay spectra for the earth.

    Note on gravity perturbation. This code supports two definitions of the
    gravity perturbation, using the keyword Q. Q=1 is simply the radial
    derivative of the perturbation of the gravtiational potential. Q=2
    corresponds to the generalized flux, which is defined 
    $$Q_2=4\pi G U_L+\frac{\ell+1}{r}\Psi+\frac{\partial \Psi}{\partial r}.$$

"""

import numpy as np

from scipy.integrate import ode, odeint

from giapy.earth_tools.viscouslove import propMatVisc_norm, gen_viscb_norm, SphericalViscSMat_norm
from giapy.earth_tools.elasticlove import propMatElas_norm, gen_elasb_norm, SphericalElasSMat_norm
from giapy.numTools.solvdeJit import solvde

class SphericalLoveVelocities(object):

    def __init__(self, params, zs, n, yEVt0=None, Q=1):
        
        # t==0 Initial guesses
        if yEVt0 is None:
            self.yEt0, self.yVt0 = np.ones((6, len(zs))), np.ones((4, len(zs)))
        else:
            self.yEt0, self.yVt0 = yEVt0

        # t>=0 Initial guesses.
        self.yE = self.yEt0.copy()
        self.yV = self.yVt0.copy()

        self.params = params
        self.zs = zs
        self.nz = len(zs)
        self.zmid = 0.5*(zs[:-1] + zs[1:])
        self.n = n
        self.Q = Q

        self.difeqElas = SphericalElasSMat_norm(n, zs, params, Q)
        self.difeqVisc = SphericalViscSMat_norm(n, zs, params, Q)

        self.indexvE = np.array([3,4,0,1,5,2])
        self.indexvV = np.array([2,3,0,1])

    def __call__(self, t, hvLv, dydt, Q=1, itmax=500, tol=1e-14, slowc=1):
    
        # Extract initial guesses, if provided, otherwise generate ones.
        #if ys is not None:
        #    yE, yV = ys
        #else:
        #    yE = np.ones((6, len(zs)))
        #    yV = np.ones((4, len(zs)))

        hv = hvLv[:self.nz]
        
        # Compute the elastic profiles 
        be = gen_elasb_norm(self.n, hv, self.params, self.zmid, self.Q)
        #difeqElas = SphericalElasSMat_norm(n, zs, params, Q=Q, b=be)
        self.difeqElas.updateProps(b=be)
        self.yE, = solvde(itmax, tol, slowc, np.ones(6), self.indexvE, 
                                3, self.yE, self.difeqElas)
    
        # Compute the viscous profiles
        bv = gen_viscb_norm(self.n, self.yE, hv, self.params, self.zmid, self.Q)
        
        #difeqVisc = SphericalViscSMat_norm(n, za, params, Q=Q, b=bv)
        self.difeqVisc.updateProps(b=bv)
        self.yV, = solvde(itmax, tol, slowc, np.ones(4), self.indexvV, 
                                2, self.yV, self.difeqVisc)

        # Extract the velocities
        dydt[:] = self.yV[[0,1],:].flatten()
    
        #return hLdv

    def solout(self):
        """Returns 
        """
        he, Le, psi, q = self.yE[[0, 1, 4, 5]]
        hdv = self.yV[0]

        return he, Le, psi, q, hdv 

class SphericalEarthOutput(object):
    """Class for organizing the recording of solution points while calculating
    viscoelastic love number relaxation in response to surface load.

    Parameters
    ----------
    times : the times to save.
    zsave : an array of radii to save the love numbers at.
    zs : the array of depths used to compute profiles at each time step
    inds : list of indices to save from relaxation objects.

    Attributes
    ----------
    times : array
        Array of times to write, in seconds.

    outArray : array (len(out.times), 6)
        Stores Ue, Uv, Ve, Vv, phi1, g1 at times.

    maxind : int
        The index of the most recent successful write step.

    Methods
    -------
    out (t, uv, f) : write surface element of uv and f.y at time t to outArray.
    converged () : fills outArray[maxind:] with outArray[maxind].
        Useful for terminating the ODE integration when a convergence criterion
        is met.
        
    """
    def __init__(self, times=None, zsave=None, zs=None, inds=None):
     
        if times is not None:
            self.times = times[:]
        else:
            self.times = np.array([0., 0.2, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 
                      10., 12., 13., 14., 15., 16., 18., 21., 25., 30., 
                      40., 50., 70., 90., 110., 130., 150.])           

        assert (zsave is not None and zs is not None) or inds is not None 

        if zsave is not None and zs is not None:
            self.inds = [np.argwhere(zs==zi)[0][0] for zi in zsave]
            self.zsave = zsave
        elif inds is not None and zs is not None:
            self.inds = inds
            self.zsave = zs[inds]
        else:
            self.inds = inds
            self.zsave = None
       
        #   he  Le  hv  Lv  psi  q  hdv
        self.outArray = np.zeros((len(self.times), len(self.inds), 9))

    def out(self, t, hv, Lv, f):
        ind = np.argwhere(self.times == t)
        try:
            self.maxind = ind[0][0]
            ind = ind[0][0]
        except IndexError:
            raise IndexError("SphericalEarthOutput received a time t={0:.3f}".format(t)+
                            " that was not in its output times.")
        he, Le, psi, q, hdv = f.solout()
        self.outArray[ind, :, 0] = he[self.inds]
        self.outArray[ind, :, 1] = hv[self.inds]
        self.outArray[ind, :, 2] = Le[self.inds]
        self.outArray[ind, :, 3] = Lv[self.inds]
        self.outArray[ind, :, 4] = psi[self.inds]
        self.outArray[ind, :, 5] = q[self.inds]
        self.outArray[ind, :, 6] = hdv[self.inds]
        self.outArray[ind, :, 7] = f.yE[2, self.inds]
        self.outArray[ind, :, 8] = f.yV[2, self.inds]



def integrateRelaxationScipy(f, out):
    """Use Scipy ode for surface response to harmonic load.

    Parameters
    ----------
    f : a function for the viscous velocities
    out : an output object
        Must have methods out.out and out.converged and data out.times.
    """
    params = f.params
    paramSurf = params(1.)
    #vislim = 1./(paramSurf['den']*paramSurf['grav']*f.alpha)
    nz = len(f.zs)

    # Get the t=0 response elastic, and save it
    f(0, np.zeros(2*nz))
    out.out(0, np.zeros(nz), np.zeros(nz), f)

    r = ode(f).set_integrator('vode', method='adams')
    #r = ode(f).set_integrator('dop853')
    r.set_initial_value(y=np.zeros(2*nz), t=0)
    timeswrite = out.times
    dts = timeswrite[1:]-timeswrite[:-1]


    for dt in dts:
        r.integrate(r.t+dt)
        out.out(r.t, r.y[:nz], r.y[nz:], f) 
