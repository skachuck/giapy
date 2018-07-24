"""
viscellove.py
Author: Samuel B. Kachuck
Date: August 1, 2017

    Compute viscoelastic decay spectra for the earth.

    Methods
    -------
    compute_viscel_numbers : Compute the viscoelastic Love numbers.

    Classes
    -------
    SphericalLoveVelocities : Class for conveniently computing velocities
    SphericalEarthOutput : Class for extracting and storing output,
        an extout object, see giapy.numTools.odeint

    Note on gravity perturbation. This code supports two definitions of the
    gravity perturbation, using the keyword Q. Q=1 is simply the radial
    derivative of the perturbation of the gravtiational potential. Q=2
    corresponds to the generalized flux, which is defined 
    $$Q_2=4\pi G U_L+\frac{\ell+1}{r}\Psi+\frac{\partial \Psi}{\partial r}.$$

"""
from __future__ import division
import numpy as np

from scipy.integrate import ode, odeint

from giapy.earth_tools.viscouslove import propMatVisc, gen_viscb, SphericalViscSMat
from giapy.earth_tools.elasticlove import propMatElas, gen_elasb, SphericalElasSMat
from giapy.numTools.solvdeJit import solvde
from giapy.numTools.odeintJit import Odeint, StepperDopr5
import giapy.numTools.odeintJit

def compute_viscel_numbers(ns, ts, zarray, params, atol=1e-4, rtol=1e-4,
                           h=1, hmin=0.001, Q=1, scaled=False, logtime=False,
                             comp=True, verbose=False):
    """
    Compute the viscoelastic Love numbers associated with params at times ts.

    Parameters
    ----------
    ns : order numbers to compute
    ts : times at which to compute
    zarray : array of depths (only length used if scaled == True)
    params : <giapy.earth_tools.earthParams.EarthParams>
        Object for storing and interpolating the earth's material parameters.
    atol, rtol : tolerances for Odeint (default 1e-4, 1e-4)
    h, hmin : initial and minimum step sizes for Odeint (default 1, 0.001)
    Q : code for gravity flux (see note above, default 1)
    scaled_time : scales the time dimension into log(t)
    comp : indicates compressibility (default True)

    Returns
    -------
    hLkt : array size (len(ns), 4, len(ts)) of love numbers
            Vertical Displacement, Horizontal Displacement, Geoid, Viscous 
    """
    
    ns = np.atleast_1d(ns)

    vels = SphericalLoveVelocities(params, zarray, ns[0], comp=comp,
                                scaled=scaled, logtime=logtime)
    # Initialize viscous Love numbers, vertical and horizontal
    hvLv0 = np.zeros(2*len(zarray)) 

    hLkt = np.zeros((len(ns), 3, len(ts)))

    # Save output times for n-dependent lithospheric acceleration.
    tets = ts.copy()
    
    for i, n in enumerate(ns):
        tau = params.tau*(n+0.5)/params.getLithFilter(n=n)
        ts = tets/tau

        # Initialize the difeq matrices for relaxation method
        vels.updateProps(n=n, z=zarray, reset_b=True)
        # Initialize the output object for the integration (inds=-1 means we
        # are looking only at the surface response).
        extout = SphericalEarthOutput(vels, ts, zs=zarray, inds=-1)

        ode = Odeint(vels, hvLv0.copy(), ts[0], ts[-1], 
                        giapy.numTools.odeintJit.StepperDopr5, atol, rtol,
                        h, hmin, xsave=ts, extout=extout)

        out = ode.integrate(verbose=verbose)
        if verbose:
            print(n, ode.h, (ode.nbad+ode.nok), ode.nbad/(ode.nbad+ode.nok))

        # Save the Love numbers for this order number.
        hLkt[i,0,:] = out.extout.outArray[:,0,0]+out.extout.outArray[:,0,1]
        hLkt[i,1,:] = out.extout.outArray[:,0,2]+out.extout.outArray[:,0,3]
        hLkt[i,2,:] = out.extout.outArray[:,0,4]
        #hLkt[i,3,:] = out.extout.outArray[:,0,1]

    # Correct n=1 case
    if ns[0] == 1:
        hLkt[0,:2,:] += hLkt[0,2,:]
        hLkt[0,2,:] -= hLkt[0,2,:]

    return np.squeeze(hLkt)

class SphericalLoveVelocities(object):
    """ Compute viscoelastic velocities at the surface of modeled earth.

   Parameters
   ----------
   params : <giapy.earth_tools.earthParams.EarthParams> object 
       for interpolation of earth parameters to relevant points.
   zs : radial mesh
   n : order number
   yEVt0 : Initial guesses for solutions. (default ones)
   Q
   comp : True (default) for compressible, False for incompressible.
   scaled : Use uniform mesh in logarithmic scaling of radial variable if True
       (default False). Transformation is chi = exp(-(rC - r)*(2n-1)/rE).
   logtime : use logarithmic time. BROKEN

   Methods
   -------
   updateProps(n, z, reset_b)
   solout()
   """
        

    def __init__(self, params, zs, n, yEVt0=None, Q=1, comp=True, 
                    scaled=False, logtime=False):
        # t==0 Initial guesses
        if yEVt0 is None:
            self.yEt0, self.yVt0 = np.ones((6, len(zs))), np.ones((4, len(zs)))
        else:
            self.yEt0, self.yVt0 = yEVt0

        # Use initial guesses without altering.
        self.yE = self.yEt0.copy()
        self.yV = self.yVt0.copy()

        self.params = params
        self.zs = zs
        self.nz = len(zs)
        self.n = n
        self.Q = Q

        # Initialize smatrices for Solvde relaxation method
        self.difeqElas = SphericalElasSMat(n, zs, params, Q, comp=comp,
                                            scaled=scaled)
        self.difeqVisc = SphericalViscSMat(n, zs, params, Q, scaled=scaled, 
                                            logtime=logtime)
        # Store between-mesh points for easier calls later.
        self.zmids = self.difeqElas.zmids

        self.indexvE = np.array([3,4,0,1,5,2])
        self.indexvV = np.array([2,3,0,1])

        self.logtime = logtime
        if logtime:
            self.tau = params.tau*(n+0.5)
        else: 
            self.tau = params.getLithFilter(n=n)

    def __call__(self, t, hvLv, dydt, itmax=500, tol=1e-14, slowc=1):
        """Compute viscous velocities given viscous displacements hvLv.

        Alters dydt in place, returns None.
        """
    
        hv = hvLv[:self.nz] 
        
        # Compute the elastic profiles 
        be = gen_elasb(self.n, hv, self.params, self.zmids, self.Q)

        self.difeqElas.updateProps(b=be)
        self.yE, = solvde(itmax, tol, slowc, np.ones(6), self.indexvE, 
                                3, self.yE, self.difeqElas)
    
        # Compute the viscous profiles
        bv = gen_viscb(self.n, self.yE, hv, self.params, self.zmids, self.Q)
        
        self.difeqVisc.updateProps(b=bv, t=t)
        self.yV, = solvde(itmax, tol, slowc, np.ones(4), self.indexvV, 
                                2, self.yV, self.difeqVisc)

        # Extract the velocities
        dydt[:] = self.yV[[0,1],:].flatten()*self.params.getLithFilter(n=self.n)


    def updateProps(self, n=None, z=None, reset_b=False):
        """Update the stored solution parameters.

        The default is to keep everything the same.

        Parameters
        ----------
        n : Update order number
        z : Update radial mesh
        reset_b : if True, set inhomogeneous vectors to zero vector
        """
        self.n = n or self.n
        self.z = self.z if z is None else z

        if self.logtime:
            self.tau = self.params.tau*(self.n+0.5)
        else:
            self.tau = self.params.getLithFilter(n=self.n) 
        
        if reset_b:
            self.difeqElas.updateProps(n=n, z=z, b=np.zeros((self.nz+1, 6)))
            self.difeqVisc.updateProps(n=n, z=z, b=np.zeros((self.nz+1, 4)))
        else:
            self.difeqElas.updateProps(n=n, z=z)
            self.difeqVisc.updateProps(n=n, z=z) 

    def solout(self):
        """Output ancillary solution variables (not direct time-deriv ones).

        Note: Used by SphericalEarthOutput at each computed time step.

        Returns
        -------
        he, Le, psi, q, hdv:
                                 [1]   [2]    [3]  [4]   [5]
            Currently stored el. vert, horiz, pot, grav, visc vels
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
    def __init__(self, f, times=None, zsave=None, zs=None, inds=None):
     
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
        self.inds = np.atleast_1d(self.inds)
        self.f = f
        self.nz = len(self.f.yE[0])
       
        #   he  hv  Le  Lv  k q  hdv f_Le    f_Lv
        self.outArray = np.zeros((len(self.times), len(self.inds), 9))

    def out(self, t, hvLv):
        ind = np.argwhere(np.abs(self.times - t)<1e-15)
        try:
            self.maxind = ind[0][0]
            ind = ind[0][0]
        except IndexError:
            raise IndexError("SphericalEarthOutput received a time t={0:.3f}".format(t)+
                            " that was not in its output times.")
        self.f(t, hvLv.copy(), 0*hvLv)
        #self.f(t, hvLv.copy())
        he, Le, k, q, hdv = self.f.solout()
        hv, Lv = hvLv[:self.nz], hvLv[self.nz:] 

        self.outArray[ind, :, 0] = he[self.inds]
        self.outArray[ind, :, 1] = hv[self.inds]
        self.outArray[ind, :, 2] = Le[self.inds]
        self.outArray[ind, :, 3] = Lv[self.inds]
        self.outArray[ind, :, 4] = k[self.inds]
        self.outArray[ind, :, 5] = q[self.inds]
        self.outArray[ind, :, 6] = hdv[self.inds]
        self.outArray[ind, :, 7] = self.f.yE[2, self.inds]
        self.outArray[ind, :, 8] = self.f.yV[2, self.inds]

